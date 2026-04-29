import torch
from dataclasses import dataclass, field
from typing import Dict, Tuple

# 8 node offsets from a voxel's grid origin (y-up convention).
# Nodes 0-3 are bottom (y=0), 4-7 are top (y=1).
#   4---5        y
#  /|  /|        |
# 7---6 |        +---x
# | 0-|-1       /
# |/  |/       z
# 3---2
NODE_OFFSETS = torch.tensor([
    [0,0,0], [1,0,0], [1,0,1], [0,0,1],
    [0,1,0], [1,1,0], [1,1,1], [0,1,1],
], dtype=torch.long)

# For each of the 6 faces, the 4 local node indices CCW from outside.
# Index matches link direction (0:-x  1:+x  2:-y  3:+y  4:-z  5:+z)
FACE_NODES = torch.tensor([
    [0,3,7,4], [1,5,6,2], [0,1,2,3],
    [4,7,6,5], [0,4,5,1], [3,2,6,7],
], dtype=torch.long)

FACE_NORMALS = torch.tensor([
    [-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1],
], dtype=torch.float32)

NEIGHBOR_OFFSETS = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]

# 28 lattice springs per voxel.
# 12 edges + 12 face diagonals + 4 body diagonals.
VOXEL_SPRINGS = torch.tensor([
    # 12 axial edges
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7],
    # 12 face diagonals
    [0,2],[1,3], [4,6],[5,7],
    [0,7],[3,4], [1,6],[2,5],
    [0,5],[1,4], [3,6],[2,7],
    # 4 body diagonals
    [0,6],[1,7],[2,4],[3,5],
], dtype=torch.long)

@dataclass
class Voxels:
    """Voxelized object (V hexahedral elements, N corner nodes, E unique lattice springs)."""
    # --- topology ---
    voxel_coords:  torch.Tensor   # (V,3) integer grid positions
    voxel_links:   torch.Tensor   # (V,6) neighbor voxel per face, -1=none
    voxel_nodes:   torch.Tensor   # (V,8) per-voxel node indices
    h:             float          # voxel edge length (world units)

    # --- dictionary for lookup ---
    coords_to_voxel: Dict[Tuple[int,int,int], int] = field(default_factory=dict, repr=False)

    # --- spring-mass system springs ---
    edges:         torch.Tensor = None   # (E,2)
    edge_rest:     torch.Tensor = None   # (E)

    # --- simulation state ---
    node_rest:     torch.Tensor = None   # (N,3)
    node_pos:      torch.Tensor = None   # (N,3)
    node_vel:      torch.Tensor = None   # (N,3)
    node_mass:     torch.Tensor = None   # (N)

    @property
    def V(self) -> int: return self.voxel_coords.shape[0]
    @property
    def N(self) -> int: return self.node_rest.shape[0]
    @property
    def E(self) -> int: return 0 if self.edges is None else self.edges.shape[0]

    @staticmethod
    def from_grid_coords(coords: torch.Tensor, h: float = 1.0) -> "Voxels":
        """Create a Voxels object from raw grid coordinates."""
        coords = coords.long()
        V = coords.shape[0]

        cmap: Dict[Tuple[int,int,int], int] = {}
        for i in range(V):
            cmap[(coords[i,0].item(), coords[i,1].item(), coords[i,2].item())] = i

        links = torch.full((V, 6), -1, dtype=torch.long)
        for i in range(V):
            cx, cy, cz = coords[i].tolist()
            for d, (dx, dy, dz) in enumerate(NEIGHBOR_OFFSETS):
                links[i, d] = cmap.get((cx+dx, cy+dy, cz+dz), -1)

        scene = Voxels(
            voxel_coords=coords, voxel_links=links, h=h,
            voxel_nodes=torch.empty(0,8,dtype=torch.long),
            node_rest=torch.empty(0,3),
            coords_to_voxel=cmap,
        )
        scene._build_nodes()
        scene._build_edges()
        return scene

    def _build_nodes(self):
        """Two voxels sharing a corner position share a node only if they are
        linked (directly or transitively at that corner). Breaking a link
        therefore duplicates the shared nodes along the fracture surface."""
        V = self.V
        node_positions = self.voxel_coords.unsqueeze(1) + NODE_OFFSETS.unsqueeze(0)  # (V,8,3)

        pos_to_nodes: Dict[Tuple[int,int,int], list] = {}
        for v in range(V):
            for n in range(8):
                key = (node_positions[v,n,0].item(),
                       node_positions[v,n,1].item(),
                       node_positions[v,n,2].item())
                pos_to_nodes.setdefault(key, []).append((v, n))

        voxel_nodes = torch.zeros(V, 8, dtype=torch.long)
        unique_node_positions: list = []
        node_idx = 0

        for pos, nodes_at_pos in pos_to_nodes.items():
            voxels_at_pos = {v for v, _ in nodes_at_pos}

            parent = {v: v for v in voxels_at_pos}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            if len(voxels_at_pos) == 1:
                for v, n in nodes_at_pos:
                    voxel_nodes[v, n] = node_idx
                unique_node_positions.append(pos)
                node_idx += 1
                continue

            for va in voxels_at_pos:
                for i in range(6):
                    vb = self.voxel_links[va, i].item()
                    if vb in voxels_at_pos:
                        ra, rb = find(va), find(vb)
                        if ra != rb:
                            parent[rb] = ra

            group_node_idx: Dict[int, int] = {}
            for v, n in nodes_at_pos:
                root = find(v)
                if root not in group_node_idx:
                    group_node_idx[root] = node_idx
                    unique_node_positions.append(pos)
                    node_idx += 1
                voxel_nodes[v, n] = group_node_idx[root]

        self.node_rest = torch.tensor(unique_node_positions, dtype=torch.float32) * self.h
        self.voxel_nodes = voxel_nodes

    def _build_edges(self):
        """Deduplicated lattice springs (28 per voxel, many shared)."""
        V = self.V
        # (V,28,2) local, expand and gather global
        s_local = VOXEL_SPRINGS.unsqueeze(0).expand(V, -1, -1)                 # (V,28,2)
        s_global = torch.gather(
            self.voxel_nodes.unsqueeze(1).expand(-1, VOXEL_SPRINGS.shape[0], -1),
            2, s_local
        )                                                                      # (V,28,2)
        pairs = s_global.reshape(-1, 2)
        pairs, _ = pairs.sort(dim=1)                # canonicalize
        self.edges = torch.unique(pairs, dim=0)
        d = self.node_rest[self.edges[:,1]] - self.node_rest[self.edges[:,0]]
        self.edge_rest = d.norm(dim=1)

    def init_state(self, density: float = 1.0):
        """Lump mass at corners; seed pos = rest, vel = 0."""
        self.node_pos = self.node_rest.clone()
        self.node_vel = torch.zeros_like(self.node_pos)
        vox_mass = density * (self.h ** 3)
        self.node_mass = torch.zeros(self.N, dtype=torch.float32)
        # each voxel distributes mass/8 to each of its 8 corners
        weights = torch.full((self.V, 8), vox_mass / 8.0)
        self.node_mass.index_add_(0, self.voxel_nodes.reshape(-1), weights.reshape(-1))

    def break_link(self, va: int, vb: int):
        for d in range(6):
            if self.voxel_links[va, d].item() == vb:
                self.voxel_links[va, d] = -1
            if self.voxel_links[vb, d].item() == va:
                self.voxel_links[vb, d] = -1

    def rebuild_after_fracture(self):
        """Rebuild node/edge arrays after break_link calls, carrying forward
        per-node state (pos, vel, mass) by copying from the old parent node
        to each of its new duplicates."""
        if self.node_pos is None:
            self._build_nodes()
            self._build_edges()
            return

        old_voxel_nodes = self.voxel_nodes.clone()
        old_pos  = self.node_pos
        old_vel  = self.node_vel
        old_mass = self.node_mass

        self._build_nodes()
        self._build_edges()

        # For every new node, find an old node at the same grid position and
        # copy its state. If a node was split (one old -> multiple new), all
        # children inherit the parent's pos/vel; mass is redistributed by the
        # number of voxels now touching each new node.
        new_pos  = torch.zeros(self.N, 3, dtype=torch.float32)
        new_vel  = torch.zeros(self.N, 3, dtype=torch.float32)
        new_mass = torch.zeros(self.N, dtype=torch.float32)
        count    = torch.zeros(self.N, dtype=torch.long)

        for v in range(self.V):
            for n in range(8):
                new_id = self.voxel_nodes[v, n].item()
                old_id = old_voxel_nodes[v, n].item()
                if count[new_id] == 0:
                    new_pos[new_id] = old_pos[old_id]
                    new_vel[new_id] = old_vel[old_id]
                count[new_id] += 1

        # Recompute per-node masses from scratch using current voxel membership.
        vox_mass = old_mass.sum() / max(self.V, 1)
        weights = torch.full((self.V, 8), vox_mass / 8.0)
        new_mass.index_add_(0, self.voxel_nodes.reshape(-1), weights.reshape(-1))

        self.node_pos, self.node_vel, self.node_mass = new_pos, new_vel, new_mass

    def connected_components(self) -> torch.Tensor:
        """Returns (V,) component labels in [0..C-1]."""
        V = self.V
        parent = list(range(V))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(V):
            for d in range(6):
                j = self.voxel_links[i, d].item()
                if j >= 0:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri
        labels = torch.tensor([find(i) for i in range(V)], dtype=torch.long)
        _, labels = torch.unique(labels, return_inverse=True)
        return labels

    def boundary_faces(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (face_node_ids, face_normals). Vertex positions are
        self.node_pos[face_node_ids] — the caller chooses rest vs current."""
        mask = self.voxel_links == -1
        voxel_idx, face_idx = mask.nonzero(as_tuple=True)
        local = FACE_NODES[face_idx]
        glob  = self.voxel_nodes[voxel_idx]
        face_nodes = glob.gather(1, local)
        normals = FACE_NORMALS[face_idx]
        return face_nodes, normals

    def boundary_face_voxels(self) -> torch.Tensor:
        mask = self.voxel_links == -1
        vi, _ = mask.nonzero(as_tuple=True)
        return vi

    def voxel_centers(self, use_pos: bool = True) -> torch.Tensor:
        """Average of each voxel's 8 corner node positions. (V,3)"""
        src = self.node_pos if (use_pos and self.node_pos is not None) else self.node_rest
        return src[self.voxel_nodes].mean(dim=1)
