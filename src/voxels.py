import torch
from dataclasses import dataclass, field
from typing import Dict, Tuple

# 8 corner offsets from a voxel's grid origin (we're using y-up convention like Minecraft :P).
# Nodes 0-3 are the bottom face (y=0) and nodes 4-7 are the top face (y=1).
#   4---5        y
#  /|  /|        |
# 7---6 |        +---x
# | 0-|-1       /
# |/  |/       z
# 3---2
CORNER_OFFSETS = torch.tensor([
    [0,0,0], [1,0,0], [1,0,1], [0,0,1],   # Bottom face (y=0)
    [0,1,0], [1,1,0], [1,1,1], [0,1,1],   # Top face    (y=1)
], dtype=torch.long)

# For each of the 6 faces, the 4 local node indices wound counter-clockwise from outside.
# Index matches link direction (0:-x  1:+x  2:-y  3:+y  4:-z  5:+z)
FACE_NODES = torch.tensor([
    [0,3,7,4],  # -x
    [1,5,6,2],  # +x
    [0,1,2,3],  # -y
    [4,7,6,5],  # +y
    [0,4,5,1],  # -z
    [3,2,6,7],  # +z
], dtype=torch.long)

FACE_NORMALS = torch.tensor([
    [-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1],
], dtype=torch.float32)

NEIGHBOR_OFFSETS = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]


@dataclass
class VoxelMesh:
    """Sparse voxel mesh with V hexahedral elements and N corner nodes."""
    # --- topology (changes on fracture) ---
    grid_coords: torch.Tensor          # (V,3) Voxel integer positions
    links:       torch.Tensor          # (V,6) Neighbor voxel per face, -1=none
    h:           float                 # Voxel edge length (world units)

    # --- FEM node arrays (rebuilt after fracture) ---
    node_rest:   torch.Tensor          # (N,3) Rest positions
    elem_nodes:  torch.Tensor          # (V,8) Per-element node indices

    # --- simulation state ---
    node_pos:    torch.Tensor          # (N,3) Current positions
    node_vel:    torch.Tensor          # (N,3) Current velocities
    node_mass:   torch.Tensor          # (N)   Lumped mass

    # --- internal lookup ---
    _cmap: Dict[Tuple[int,int,int],int] = field(default_factory=dict, repr=False)

    @property
    def V(self) -> int:
        return self.grid_coords.shape[0]

    @property
    def N(self) -> int:
        return self.node_pos.shape[0]

    @staticmethod
    def from_grid_coords(coords: torch.Tensor, h: float = 1.0, density: float = 1.0) -> "VoxelMesh":
        """Build a VoxelMesh from integer grid coordinates."""
        coords = coords.long()
        V = coords.shape[0]

        # 3D coordinate to voxel-index map
        cmap: Dict[Tuple[int,int,int],int] = {}
        for i in range(V):
            cmap[(coords[i,0].item(), coords[i,1].item(), coords[i,2].item())] = i

        # Face adjacency links
        links = torch.full((V, 6), -1, dtype=torch.long)
        for i in range(V):
            cx, cy, cz = coords[i].tolist()
            for d, (dx, dy, dz) in enumerate(NEIGHBOR_OFFSETS):
                j = cmap.get((cx+dx, cy+dy, cz+dz), -1)
                links[i, d] = j

        mesh = VoxelMesh(
            grid_coords=coords, links=links, h=h,
            node_rest=torch.empty(0,3), elem_nodes=torch.empty(0,8,dtype=torch.long),
            node_pos=torch.empty(0,3), node_vel=torch.empty(0,3),
            node_mass=torch.empty(0), _cmap=cmap,
        )
        mesh._build_nodes(density)
        return mesh

    def _build_nodes(self, density: float = 1.0):
        """(Re)compute node arrays from current voxel + link topology.

        Two voxels sharing a corner position share a node only if
        they are connected by links (directly or transitively through
        other voxels at that same corner).  Breaking a link therefore
        duplicates the shared nodes along the fracture surface.
        """
        V = self.V
        gc = self.grid_coords                                    # (V,3) long
        corners = gc.unsqueeze(1) + CORNER_OFFSETS.unsqueeze(0)  # (V,8,3)

        # Group (voxel, local_corner) by grid position of the corner
        pos_to_vc: Dict[Tuple[int,int,int], list] = {}
        for v in range(V):
            for c in range(8):
                key = (corners[v,c,0].item(), corners[v,c,1].item(), corners[v,c,2].item())
                pos_to_vc.setdefault(key, []).append((v, c))

        elem_nodes = torch.zeros(V, 8, dtype=torch.long)
        node_grid_positions: list = []   # grid-int coords of each node
        nid = 0

        for pos_key, vc_list in pos_to_vc.items():
            voxels_here = {v for v, _ in vc_list}

            if len(voxels_here) == 1:
                for v, c in vc_list:
                    elem_nodes[v, c] = nid
                node_grid_positions.append(pos_key)
                nid += 1
                continue

            # Union-find (merge voxels at this corner that are still linked)
            par = {v: v for v in voxels_here}
            def find(x):
                while par[x] != x:
                    par[x] = par[par[x]]
                    x = par[x]
                return x

            for v in voxels_here:
                for d in range(6):
                    j = self.links[v, d].item()
                    if j in voxels_here:
                        ra, rb = find(v), find(j)
                        if ra != rb:
                            par[rb] = ra

            # One node per connected group
            group_nid: Dict[int, int] = {}
            for v, c in vc_list:
                root = find(v)
                if root not in group_nid:
                    group_nid[root] = nid
                    node_grid_positions.append(pos_key)
                    nid += 1
                elem_nodes[v, c] = group_nid[root]

        N = len(node_grid_positions)
        self.node_rest  = torch.tensor(node_grid_positions, dtype=torch.float32) * self.h
        self.elem_nodes = elem_nodes
        self.node_pos  = self.node_rest.clone()
        self.node_vel  = torch.zeros(N, 3)

        # lumped mass (voxel mass split equally to 8 corners)
        vmass = density * self.h ** 3
        self.node_mass = torch.zeros(N)
        for v in range(V):
            for c in range(8):
                self.node_mass[elem_nodes[v, c]] += vmass / 8.0

    def break_link(self, va: int, vb: int):
        """Voxel fracturing. Severs face-adjacency between two voxels."""
        for d in range(6):
            if self.links[va, d].item() == vb:
                self.links[va, d] = -1
            if self.links[vb, d].item() == va:
                self.links[vb, d] = -1

    def rebuild_after_fracture(self, density: float = 1.0):
        """Rebuild nodes. Copies state from old node arrays."""
        old_pos  = self.node_pos.clone()
        old_vel  = self.node_vel.clone()
        old_elem = self.elem_nodes.clone()

        self._build_nodes(density)

        for v in range(self.V):
            for c in range(8):
                self.node_pos[self.elem_nodes[v, c]] = old_pos[old_elem[v, c]]
                self.node_vel[self.elem_nodes[v, c]] = old_vel[old_elem[v, c]]

    def connected_components(self) -> torch.Tensor:
        """Union-find on voxel links. Returns (V) labels in [0..C-1]."""
        V = self.V
        par = list(range(V))
        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for i in range(V):
            for d in range(6):
                j = self.links[i, d].item()
                if j >= 0:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        par[rj] = ri
        labels = torch.tensor([find(i) for i in range(V)], dtype=torch.long)
        _, labels = torch.unique(labels, return_inverse=True)
        return labels

    def boundary_faces(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exposed boundary quad faces (i.e. those that are part of exactly one voxel)."""
        mask = self.links == -1               # (V,6)
        vi, fi = mask.nonzero(as_tuple=True)

        local   = FACE_NODES[fi]              # (F,4) local corner ids
        glob    = self.elem_nodes[vi]         # (F,8) global node ids
        fnodes  = glob.gather(1, local)       # (F,4)
        verts   = self.node_pos[fnodes]       # (F,4,3)
        normals = FACE_NORMALS[fi]            # (F,3)
        return verts, normals

    def boundary_face_voxels(self) -> torch.Tensor:
        """Which voxel owns each boundary face. Same ordering as `boundary_faces()`."""
        mask = self.links == -1
        vi, _ = mask.nonzero(as_tuple=True)
        return vi

    @staticmethod
    def cube(side: int, h: float = 1.0, density: float = 1.0) -> "VoxelMesh":
        """Solid cube of side^3 voxels."""
        r = torch.arange(side)
        gx, gy, gz = torch.meshgrid(r, r, r, indexing="ij")
        coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
        return VoxelMesh.from_grid_coords(coords, h=h, density=density)

    @staticmethod
    def sphere(radius: int, h: float = 1.0, density: float = 1.0) -> "VoxelMesh":
        """Solid sphere of given radius (in voxels)."""
        r = torch.arange(-radius, radius + 1)
        gx, gy, gz = torch.meshgrid(r, r, r, indexing="ij")
        mask = gx**2 + gy**2 + gz**2 <= radius**2
        coords = torch.stack([gx[mask], gy[mask], gz[mask]], dim=-1)
        return VoxelMesh.from_grid_coords(coords, h=h, density=density)

    @staticmethod
    def from_file(path: str, h: float = 1.0, density: float = 1.0) -> "VoxelMesh":
        """Load from a text file with one `x y z` line per voxel."""
        coords = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                x, y, z = line.split()
                coords.append([int(x), int(y), int(z)])
        return VoxelMesh.from_grid_coords(torch.tensor(coords), h=h, density=density)