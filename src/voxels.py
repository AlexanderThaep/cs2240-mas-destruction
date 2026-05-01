from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Dict, Tuple

from mesh import Mesh

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

    # topology
    voxel_coords:  torch.Tensor     # (V,3) integer grid positions
    voxel_links:   torch.Tensor     # (V,6) neighbor voxel per face, -1=none
    voxel_nodes:   torch.Tensor     # (V,8) per-voxel node indices
    voxel_length:  float            # voxel edge length (world units)

    # spring-mass system springs
    edges_rest:    torch.Tensor     # (E)
    edges:         torch.Tensor     # (E,2)

    # simulation state
    nodes_rest:     torch.Tensor    # (N,3) node positions at rest
    nodes_pos:      torch.Tensor    # (N,3) node positions during simulation
    nodes_vel:      torch.Tensor    # (N,3) node velocities during simulation
    nodes_mass:     torch.Tensor    # (N)

    @property
    def V(self) -> int: return self.voxel_coords.shape[0]
    @property
    def N(self) -> int: return self.nodes_rest.shape[0]
    @property
    def E(self) -> int: return 0 if self.edges is None else self.edges.shape[0]

    def _init_state(self, density: float = 1.0):
        """Lump mass at corners; seed pos = rest, vel = 0."""
        self.nodes_pos = self.nodes_rest.clone()
        self.nodes_vel = torch.zeros_like(self.nodes_pos)
        vox_mass = density * (self.voxel_length ** 3)
        self.nodes_mass = torch.zeros(self.N, dtype=torch.float32)
        # each voxel distributes mass/8 to each of its 8 corners
        weights = torch.full((self.V, 8), vox_mass / 8.0)
        self.nodes_mass.index_add_(0, self.voxel_nodes.reshape(-1), weights.reshape(-1))

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
        pairs, _ = pairs.sort(dim=1)                                           # canonicalize
        self.edges = torch.unique(pairs, dim=0)
        d = self.nodes_rest[self.edges[:,1]] - self.nodes_rest[self.edges[:,0]]
        self.edges_rest = d.norm(dim=1)

    def __init__(
        self,
        voxel_coords: torch.Tensor,   # (V,3)
        voxel_links: torch.Tensor,    # (V,6)
        voxel_nodes: torch.Tensor,    # (V,8)
        voxel_length: float,
        nodes_rest: torch.Tensor      # (N,3)
    ):
        self.voxel_coords = voxel_coords
        self.voxel_links  = voxel_links
        self.voxel_length = voxel_length

        self.voxel_nodes  = voxel_nodes
        self.nodes_rest = nodes_rest

        self._build_edges()
        self._init_state()

    @staticmethod
    def from_meshes(meshes: list[Mesh], h: float = 1.0) -> Voxels:
        """Create a Voxels object from a list of meshes"""

        voxel_offset = 0
        node_offset = 0

        # topology accumulators
        global_positions = []
        global_links = []
        global_nodes = []

        # sim state accumulators
        global_nodes_rest = []

        # construct voxels, links, and nodes
        for id, m in enumerate(meshes):
            n = m.voxelmap.shape[0]

            pos_to_index: Dict[Tuple[int,int,int], int] = {}
            pos_to_nodes: Dict[Tuple[int,int,int], list] = {}

            positions = m.voxelmap[:, :3]
            links = torch.full((n, 6), -1, dtype=torch.long)
            nodes = torch.zeros(n, 8, dtype=torch.long)

            for i in range(n):
                x, y, z = positions[i].tolist()
                pos_to_index[(x, y, z)] = i

            for i in range(n):
                cx, cy, cz = positions[i].tolist()
                for d, (dx, dy, dz) in enumerate(NEIGHBOR_OFFSETS):
                    links[i, d] = pos_to_index.get((cx+dx, cy+dy, cz+dz), -1)

            node_positions = positions.unsqueeze(1) + NODE_OFFSETS.unsqueeze(0)
            for i in range(n):
                for j in range(8):
                    key = (
                        int(node_positions[i, j, 0].item()),
                        int(node_positions[i, j, 1].item()),
                        int(node_positions[i, j, 2].item()))
                    pos_to_nodes.setdefault(key, []).append((i, j))

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
                    for v, m in nodes_at_pos:
                        nodes[v, m] = node_idx
                    unique_node_positions.append(pos)
                    node_idx += 1
                    continue

                for va in voxels_at_pos:
                    for i in range(6):
                        vb = links[va, i].item()
                        if vb in voxels_at_pos:
                            ra, rb = find(va), find(vb)
                            if ra != rb:
                                parent[rb] = ra

                group_node_idx: Dict[int, int] = {}
                for v, m in nodes_at_pos:
                    root = find(v)
                    if root not in group_node_idx:
                        group_node_idx[root] = node_idx
                        unique_node_positions.append(pos)
                        node_idx += 1
                    nodes[v, m] = group_node_idx[root]

            nodes_rest = torch.tensor(unique_node_positions, dtype=torch.float32) * h

            links[links >= 0] += voxel_offset
            nodes[nodes >= 0] += node_offset

            global_positions.append(positions)
            global_links.append(links)
            global_nodes.append(nodes)
            global_nodes_rest.append(nodes_rest)

            voxel_offset += n
            node_offset += nodes_rest.shape[0]

        coords = torch.cat(global_positions, dim=0)
        links = torch.cat(global_links, dim=0)
        nodes = torch.cat(global_nodes, dim=0)
        nodes_rest = torch.cat(global_nodes_rest, dim=0)

        return Voxels(
            voxel_coords=coords,
            voxel_links=links,
            voxel_nodes=nodes,
            voxel_length=h,
            nodes_rest=nodes_rest
        )
            
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