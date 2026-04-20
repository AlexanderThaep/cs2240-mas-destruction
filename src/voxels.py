import torch
from dataclasses import dataclass, field
from typing import Dict, Tuple

# 8 node offsets from a voxel's grid origin (we're using y-up convention like Minecraft :P).
# Nodes 0-3 are the bottom face (y=0) and nodes 4-7 are the top face (y=1).
#   4---5        y
#  /|  /|        |
# 7---6 |        +---x
# | 0-|-1       /
# |/  |/       z
# 3---2
NODE_OFFSETS = torch.tensor([
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
    voxel_coords:  torch.Tensor  # (V,3) Voxel grid integer positions
    links:         torch.Tensor  # (V,6) Neighbor voxel per face, -1=none
    h:             float         # Voxel edge length (world units)

    # --- voxel node arrays rebuilt after fracture ---
    voxel_nodes:   torch.Tensor  # (V,8) Per-voxel node indices
    node_rest:     torch.Tensor  # (N,3) Rest positions

    # --- simulation state ---
    node_pos:      torch.Tensor  # (N,3) Current positions
    node_vel:      torch.Tensor  # (N,3) Current velocities
    node_mass:     torch.Tensor  # (N)   Lumped mass

    # --- internal lookup ---
    _cmap: Dict[Tuple[int,int,int],int] = field(default_factory=dict, repr=False)

    @property
    def V(self) -> int:
        return self.voxel_coords.shape[0]

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
            voxel_coords=coords, links=links, h=h,
            node_rest=torch.empty(0,3), voxel_nodes=torch.empty(0,8,dtype=torch.long),
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
        grid_coords = self.voxel_coords                                      # (V,3)
        node_positions = grid_coords.unsqueeze(1) + NODE_OFFSETS.unsqueeze(0)  # (V,8,3)

        # Group (voxel, node) by grid position of the node, so that pos_to_node[p]
        # contains up to 8 voxels that have that grid-space as a valid node.
        pos_to_nodes: Dict[Tuple[int,int,int], list] = {}
        for v in range(V):
            for n in range(8):
                key = (node_positions[v,n,0].item(), node_positions[v,n,1].item(), node_positions[v,n,2].item())
                pos_to_nodes.setdefault(key, []).append((v, n))

        voxel_nodes = torch.zeros(V, 8, dtype=torch.long)
        unique_node_positions: list = []
        node_idx = 0

        for pos, nodes_at_pos in pos_to_nodes.items():
            voxels_at_pos = {v for v, _ in nodes_at_pos}

            # Union-find function to merge voxels at this node that are linked
            parent = {v: v for v in voxels_at_pos}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            # Check if there's only one voxel at this node position (no need to union then)
            if len(voxels_at_pos) == 1:
                for v, n in nodes_at_pos:
                    voxel_nodes[v, n] = node_idx
                unique_node_positions.append(pos)
                node_idx += 1
                continue

            # Union linked voxels at this node
            for va in voxels_at_pos:
                for i in range(6):
                    vb = self.links[va, i].item()
                    if vb in voxels_at_pos:
                        ra, rb = find(va), find(vb)
                        if ra != rb:
                            parent[rb] = ra

            # Enforce one node per connected group
            group_node_idx: Dict[int, int] = {}
            for v, n in nodes_at_pos:
                root = find(v)
                if root not in group_node_idx:
                    group_node_idx[root] = node_idx
                    unique_node_positions.append(pos)
                    node_idx += 1
                voxel_nodes[v, n] = group_node_idx[root]

        N = len(unique_node_positions)
        self.node_rest  = torch.tensor(unique_node_positions, dtype=torch.float32) * self.h
        self.voxel_nodes = voxel_nodes
        self.node_pos  = self.node_rest.clone()
        self.node_vel  = torch.zeros(N, 3)

        # lumped mass (voxel mass distributed equally to 8 corners)
        voxel_mass = density * self.h ** 3
        self.node_mass = torch.zeros(N)
        for v in range(V):
            for n in range(8):
                self.node_mass[voxel_nodes[v, n]] += voxel_mass / 8.0

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
        old_nodes = self.voxel_nodes.clone()

        self._build_nodes(density)

        for v in range(self.V):
            for n in range(8):
                self.node_pos[self.voxel_nodes[v, n]] = old_pos[old_nodes[v, n]]
                self.node_vel[self.voxel_nodes[v, n]] = old_vel[old_nodes[v, n]]

    def connected_components(self) -> torch.Tensor:
        """Union-find on voxel links. Returns (V) labels in [0..C-1]."""
        V = self.V
        parent = list(range(V))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(V):
            for d in range(6):
                j = self.links[i, d].item()
                if j >= 0:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri
        labels = torch.tensor([find(i) for i in range(V)], dtype=torch.long)
        _, labels = torch.unique(labels, return_inverse=True)
        return labels

    def boundary_faces(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exposed boundary quad faces (i.e. those that are part of exactly one voxel)."""
        mask = self.links == -1               # (V,6)
        vi, fi = mask.nonzero(as_tuple=True)

        local   = FACE_NODES[fi]              # (F,4) local corner ids
        glob    = self.voxel_nodes[vi]        # (F,8) global node ids
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