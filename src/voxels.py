import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as scipy_cc

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

NEIGHBOR_OFFSETS = torch.tensor([
    [-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1],
], dtype=torch.long)

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


def _compute_face_pairs() -> Tensor:
    """Per-direction (local_a, local_b) corner pairs unified by a face link."""
    pairs = torch.zeros(6, 4, 2, dtype=torch.long)
    for d in range(6):
        opposite = d ^ 1  # 0<->1, 2<->3, 4<->5
        a_locals = FACE_NODES[d]
        b_locals = FACE_NODES[opposite]
        a_world = NODE_OFFSETS[a_locals]
        b_world = NODE_OFFSETS[b_locals] + NEIGHBOR_OFFSETS[d]
        for i in range(4):
            for j in range(4):
                if torch.equal(a_world[i], b_world[j]):
                    pairs[d, i, 0] = a_locals[i]
                    pairs[d, i, 1] = b_locals[j]
                    break
    return pairs

FACE_PAIRS = _compute_face_pairs()  # (6, 4, 2)


def _undirected_components(n: int, src: Tensor, dst: Tensor) -> Tensor:
    """Component labels in [0, C) for an undirected graph on n nodes."""
    if src.numel() == 0:
        return torch.arange(n, dtype=torch.long)
    rows = np.concatenate([src.numpy(), dst.numpy()])
    cols = np.concatenate([dst.numpy(), src.numpy()])
    data = np.ones(rows.shape[0], dtype=np.int8)
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    _, labels = scipy_cc(adj, directed=False, return_labels=True)
    return torch.from_numpy(labels.astype(np.int64))


@dataclass
class Voxels:
    """Voxelized object (V hexahedral elements, N corner nodes, E unique lattice springs)."""
    # topology
    voxel_coords:    Tensor         # (V,3) integer grid positions
    voxel_links:     Tensor         # (V,6) neighbor voxel per face, -1=none
    voxel_nodes:     Tensor         # (V,8) per-voxel node indices
    h:               float          # voxel edge length (world units)

    # spring-mass system springs
    edges:           Tensor = None  # (E,2)
    edge_lens_rest:  Tensor = None  # (E)

    # simulation state
    node_rest:       Tensor = None  # (N,3)
    node_pos:        Tensor = None  # (N,3)
    node_vel:        Tensor = None  # (N,3)
    node_mass:       Tensor = None  # (N)

    @property
    def V(self) -> int: return self.voxel_coords.shape[0]
    @property
    def N(self) -> int: return self.node_rest.shape[0]
    @property
    def E(self) -> int: return 0 if self.edges is None else self.edges.shape[0]

    @staticmethod
    def from_grid_coords(coords: Tensor, h: float = 1.0) -> "Voxels":
        """Create a Voxels object from raw grid coordinates."""
        coords = coords.long()
        V = coords.shape[0]

        if V == 0:
            return Voxels(
                voxel_coords=coords,
                voxel_links=torch.empty(0, 6, dtype=torch.long),
                voxel_nodes=torch.empty(0, 8, dtype=torch.long),
                node_rest=torch.empty(0, 3),
                edges=torch.empty(0, 2, dtype=torch.long),
                edge_lens_rest=torch.empty(0),
                h=h,
            )

        cmin = coords.min(dim=0).values
        cmax = coords.max(dim=0).values
        cshift = coords - cmin + 1  # +1 leaves room for -1 neighbor offset
        extent = (cmax - cmin + 3).long()
        s_yz = extent[1] * extent[2]
        s_z  = extent[2]

        keys = cshift[:, 0] * s_yz + cshift[:, 1] * s_z + cshift[:, 2]  # (V,)
        sort_idx = keys.argsort()
        sorted_keys = keys[sort_idx]

        nbr = cshift.unsqueeze(1) + NEIGHBOR_OFFSETS.unsqueeze(0)        # (V,6,3)
        nbr_keys = (nbr[..., 0] * s_yz + nbr[..., 1] * s_z + nbr[..., 2]).reshape(-1)
        lo = torch.searchsorted(sorted_keys, nbr_keys).clamp(max=V-1)
        hit = sorted_keys[lo] == nbr_keys
        links = torch.where(hit, sort_idx[lo], torch.tensor(-1, dtype=torch.long)).reshape(V, 6)

        voxels = Voxels(
            voxel_coords=coords,
            voxel_links=links,
            voxel_nodes=torch.empty(0, 8, dtype=torch.long),
            node_rest=torch.empty(0, 3),
            h=h,
        )
        voxels._build_nodes()
        voxels._build_edges()
        return voxels

    @staticmethod
    def from_meshes(meshes: List[Mesh], h: float = 1.0) -> Tuple["Voxels", List[Tensor]]:
        """Combine several Mesh objects into one Voxels grid."""
        coords_list = [m.voxelmap[:, :3].long() for m in meshes]
        bounds = [0]
        for c in coords_list:
            bounds.append(bounds[-1] + c.shape[0])
        coords = torch.cat(coords_list, dim=0)
        voxels = Voxels.from_grid_coords(coords, h=h)

        node_groups = [
            voxels.voxel_nodes[bounds[i]:bounds[i+1]].reshape(-1).unique()
            for i in range(len(meshes))
        ]
        return voxels, node_groups

    def _build_nodes(self):
        """Two voxels sharing a corner position share a node only if they are
        linked (directly or transitively at that corner). Breaking a link
        therefore duplicates the shared nodes along the fracture surface."""
        V = self.V
        if V == 0:
            self.node_rest = torch.empty(0, 3)
            self.voxel_nodes = torch.empty(0, 8, dtype=torch.long)
            return

        valid = self.voxel_links >= 0
        va, d = valid.nonzero(as_tuple=True)
        vb = self.voxel_links[va, d]
        keep = va < vb  # process each undirected link once
        va, d, vb = va[keep], d[keep], vb[keep]

        merge = FACE_PAIRS[d]                                           # (L,4,2)
        ids_a = (va.unsqueeze(-1) * 8 + merge[..., 0]).reshape(-1)      # (4L,)
        ids_b = (vb.unsqueeze(-1) * 8 + merge[..., 1]).reshape(-1)      # (4L,)

        labels = _undirected_components(V * 8, ids_a, ids_b)            # (8V,)
        self.voxel_nodes = labels.reshape(V, 8)

        # World position per (v, local_n); take the first occurrence per label as canonical
        node_world = (self.voxel_coords.unsqueeze(1) + NODE_OFFSETS.unsqueeze(0)).reshape(V*8, 3)
        labels_np = labels.numpy()
        sort_idx = labels_np.argsort(kind="stable")
        sorted_labels = labels_np[sort_idx]
        first = np.empty_like(sorted_labels, dtype=bool)
        first[0] = True
        first[1:] = sorted_labels[1:] != sorted_labels[:-1]
        rep_indices = sort_idx[first]                                   # one rep per label
        self.node_rest = node_world[torch.from_numpy(rep_indices)].float() * self.h

    def _build_edges(self):
        """Deduplicated lattice springs (28 per voxel, many shared)."""
        V = self.V
        if V == 0:
            self.edges = torch.empty(0, 2, dtype=torch.long)
            self.edge_lens_rest = torch.empty(0)
            return
        s_local = VOXEL_SPRINGS.unsqueeze(0).expand(V, -1, -1)                 # (V,28,2)
        s_global = torch.gather(
            self.voxel_nodes.unsqueeze(1).expand(-1, VOXEL_SPRINGS.shape[0], -1),
            2, s_local
        )                                                                      # (V,28,2)
        pairs = s_global.reshape(-1, 2)
        pairs, _ = pairs.sort(dim=1)                # canonicalize
        self.edges = torch.unique(pairs, dim=0)
        d = self.node_rest[self.edges[:,1]] - self.node_rest[self.edges[:,0]]
        self.edge_lens_rest = d.norm(dim=1)

    def voxel_rotations(self, pos: Tensor, indices: Tensor = None) -> Tensor:
        """Per-voxel rotation matrices."""
        vn = self.voxel_nodes if indices is None else self.voxel_nodes[indices]  # (K,8)
        cur = pos[vn]                                                            # (K,8,3)
        rest = self.node_rest[vn]                                                # (K,8,3)
        cur_c = cur - cur.mean(dim=1, keepdim=True)
        rest_c = rest - rest.mean(dim=1, keepdim=True)
        A = cur_c.transpose(-1, -2) @ rest_c                                     # (K,3,3)
        U, _, Vh = torch.linalg.svd(A)
        det_sign = torch.linalg.det(U) * torch.linalg.det(Vh)                    # (K) +/- 1
        diag = torch.ones_like(A[..., 0])                                        # (K,3)
        diag[..., 2] = det_sign
        return (U * diag.unsqueeze(-2)) @ Vh                                     # (K,3,3)

    def init_state(self, density: float = 1.0):
        """Lump mass at corners; seed pos = rest, vel = 0."""
        self.node_pos = self.node_rest.clone()
        self.node_vel = torch.zeros_like(self.node_pos)
        vox_mass = density * (self.h ** 3)
        self.node_mass = torch.zeros(self.N, dtype=torch.float32)
        # each voxel distributes mass/8 to each of its 8 corners
        weights = torch.full((self.V, 8), vox_mass / 8.0)
        self.node_mass.index_add_(0, self.voxel_nodes.reshape(-1), weights.reshape(-1))

    def break_links(self, pairs: Tensor):
        """Sever a batch of voxel-voxel links given as (K,2) pairs."""
        if pairs.numel() == 0:
            return
        va, vb = pairs[:, 0], pairs[:, 1]
        match_a = self.voxel_links[va] == vb.unsqueeze(-1)  # (K,6)
        match_b = self.voxel_links[vb] == va.unsqueeze(-1)  # (K,6)
        k_a, d_a = match_a.nonzero(as_tuple=True)
        k_b, d_b = match_b.nonzero(as_tuple=True)
        self.voxel_links[va[k_a], d_a] = -1
        self.voxel_links[vb[k_b], d_b] = -1

    def rebuild_after_fracture(self):
        """Rebuild node/edge arrays after break_links calls, carrying forward
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

        new_pos  = torch.zeros(self.N, 3, dtype=torch.float32)
        new_vel  = torch.zeros(self.N, 3, dtype=torch.float32)
        new_mass = torch.zeros(self.N, dtype=torch.float32)

        # Bulk-copy: all (v,n) sharing a new node id share the same world pos.
        new_flat = self.voxel_nodes.reshape(-1)
        old_flat = old_voxel_nodes.reshape(-1)
        new_pos[new_flat] = old_pos[old_flat]
        new_vel[new_flat] = old_vel[old_flat]

        # Recompute per-node masses from scratch using current voxel membership.
        vox_mass = old_mass.sum() / max(self.V, 1)
        weights = torch.full((self.V, 8), vox_mass / 8.0)
        new_mass.index_add_(0, new_flat, weights.reshape(-1))

        self.node_pos, self.node_vel, self.node_mass = new_pos, new_vel, new_mass

    def connected_components(self) -> Tensor:
        """Returns (V,) component labels in [0..C-1]."""
        V = self.V
        if V == 0:
            return torch.empty(0, dtype=torch.long)
        valid = self.voxel_links >= 0
        va, d = valid.nonzero(as_tuple=True)
        vb = self.voxel_links[va, d]
        return _undirected_components(V, va, vb)

    def boundary_faces(self) -> Tuple[Tensor, Tensor]:
        """Returns (face_node_ids, face_normals). Vertex positions are
        self.node_pos[face_node_ids] — the caller chooses rest vs current."""
        mask = self.voxel_links == -1
        voxel_idx, face_idx = mask.nonzero(as_tuple=True)
        local = FACE_NODES[face_idx]
        glob  = self.voxel_nodes[voxel_idx]
        face_nodes = glob.gather(1, local)
        normals = FACE_NORMALS[face_idx]
        return face_nodes, normals

    def boundary_face_voxels(self) -> Tensor:
        mask = self.voxel_links == -1
        vi, _ = mask.nonzero(as_tuple=True)
        return vi

    def voxel_centers(self, use_pos: bool = True) -> Tensor:
        """Average of each voxel's 8 corner node positions. (V,3)"""
        src = self.node_pos if (use_pos and self.node_pos is not None) else self.node_rest
        return src[self.voxel_nodes].mean(dim=1)
