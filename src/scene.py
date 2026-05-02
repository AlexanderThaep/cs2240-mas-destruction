import torch
from torch import Tensor
from voxels import Voxels
from dataclasses import dataclass
from scipy.sparse.linalg import LinearOperator, cg
import numpy as np

# TODO: add a guard in a single location to ensure that there is >= 1 voxel.
# TODO: rename Scene to Simulation (and use 'sim' instead of scene)

@dataclass
class Scene:
    # simulation parameters
    voxels:   Voxels                             # voxels in scene
    k:        float                              # spring constant (stiffness)
    dt:       float                              # Euler approximation timestep
    gravity:  Tensor = Tensor([0.0, -9.8, 0.0])  # downward gravitational acceleration

    # collisions
    ground_y:        float = 0.0    # height of ground plane
    ground_k:        float = 1e4    # ground penalty stiffness
    ground_c:        float = 2.0    # ground velocity damping
    self_collide:    bool  = False  # voxel-voxel inter-component collisions
    self_collide_k:  float = 1e3    # self-collision penalty stiffness

    # caches
    edge_lens:  Tensor = None
    edge_dirs:  Tensor = None

    def refresh_edges(self, pos: Tensor):
        edges = self.voxels.edges
        edge_vecs = pos[edges[:, 1]] - pos[edges[:, 0]]
        edge_lens = edge_vecs.norm(dim=1).clamp(min=1e-10)
        self.edge_lens = edge_lens
        self.edge_dirs = edge_vecs / edge_lens.unsqueeze(-1)

    def ground_forces(self, pos: Tensor) -> Tensor:
        """Normal force felt by nodes hitting the ground."""
        penalty = (self.ground_y - pos[:, 1]).clamp(min=0.0)                    # (N)
        in_contact = penalty > 0                                                # (N)
        node_vel_down = self.voxels.node_vel[:, 1].clamp(max=0.0) * in_contact  # (N)
        force = torch.zeros_like(pos)                                           # (N,3)
        force[:, 1] = self.ground_k * penalty - self.ground_c * node_vel_down   # (N)
        return force

    def collision_candidates(self, pos: Tensor) -> tuple[Tensor, Tensor]:
        """Returns candidate voxel pairs that could be colliding based on spatial locality."""
        h = self.voxels.h
        nodes = self.voxels.voxel_nodes                                                  # (V,8)
        V = nodes.shape[0]

        centroids = pos[nodes].mean(dim=1)                                                 # (V,3)
        cells = (centroids / h).floor().long()                                             # (V,3)
        origin = cells.min(dim=0).values - 1                                             # (3)
        keys = morton_code(cells - origin)                                               # (V)
        keys = keys[keys.argsort()]                                                      # voxels on Z-order curve

        # In 3D each cell is in a 3x3x3 = 27 cell neighborhood
        offsets = torch.cartesian_prod(*[torch.arange(-1, 2)] * 3)                       # (27,3)
        neighbor_keys = morton_code(cells.unsqueeze(1) + offsets.unsqueeze(0) - origin)  # (V,27)
        low = torch.searchsorted(keys, neighbor_keys.reshape(-1), right=False)           # (27V)
        high = torch.searchsorted(keys, neighbor_keys.reshape(-1), right=True)           # (27V)
        count = high - low                                                               # (27V)
        total = int(count.sum())

        if total == 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty

        query_voxel = torch.arange(V).repeat_interleave(27)               # (27V)
        a = query_voxel.repeat_interleave(count)                          # (total)
        range_start = low.repeat_interleave(count)                        # (total)
        query_offset = torch.cumsum(count, dim=0) - count
        pair_offset = query_offset.repeat_interleave(count)               # (total)
        offset = torch.arange(total) - pair_offset
        b = sort_idx[range_start + offset]                                # (total)

        keep = a < b
        return a[keep], b[keep]

    def collision_forces(self, pos: Tensor) -> Tensor:
        """Collision force felt by voxels hitting other voxels."""
        h = self.voxels.h
        nodes = self.voxels.voxel_nodes
        a, b = self.collision_candidates(pos)

        if a.numel() == 0:
            return torch.zeros_like(pos)

        components = self.voxels.connected_components()
        cross = components[a] != components[b]
        a, b = a[cross], b[cross]

        if a.numel() == 0:
            return torch.zeros_like(pos)

        # Actual centroid distance for each candidate pair
        centroids = pos[nodes].mean(dim=1)         # (V,3)
        diff = centroids[b] - centroids[a]         # (K,3)
        dist = diff.norm(dim=-1).clamp(min=1e-10)  # (K)
        penalty = (h - dist).clamp(min=0.0)        # (K)
        contact = penalty > 0

        if not contact.any():
            return torch.zeros_like(pos)

        a, b, penalty, dist, diff = a[contact], b[contact], penalty[contact], dist[contact], diff[contact]
        voxel_push = (self.self_collide_k * penalty / dist).unsqueeze(-1) * diff  # (K,3) on b from a
        node_push = (voxel_push / 8.0).repeat_interleave(8, dim=0)                # (8K, 3)
        force = torch.zeros_like(pos)
        force.index_add_(0, nodes[b].reshape(-1),  node_push)
        force.index_add_(0, nodes[a].reshape(-1), -node_push)
        return force

    def external_forces(self, pos: Tensor) -> Tensor:
        """External forces felt by each node (gravity and collisions)"""
        gravity = self.voxels.node_mass.unsqueeze(-1) * self.gravity.unsqueeze(0)
        forces = gravity + self.ground_forces(pos) + self.collision_forces(pos)
        return forces

    def internal_forces(self, pos: Tensor) -> Tensor:
        """Internal spring forces felt by each node (sum over Hooke's Law spring forces)."""
        forces = torch.zeros_like(pos)  # (N,3)
        edges = self.voxels.edges                                     # (E,2)
        L, L0 = self.edge_lens, self.voxels.edge_lens_rest            # (E)

        force_i = (self.k * (L - L0)).unsqueeze(-1) * self.edge_dirs  # (E,3)

        forces.index_add_(0, edges[:, 0],  force_i)
        forces.index_add_(0, edges[:, 1], -force_i)
        return forces

    def lhs_Ax(self, x: Tensor) -> Tensor:
        """LHS of system Ax = b in Wu et al. [2022, Section 3.1] used in PCG solver."""
        Mx = self.voxels.node_mass.unsqueeze(-1) * x              # (N,1)
        edges = self.voxels.edges                                 # (E,2)
        x_diff = x[edges[:, 0]] - x[edges[:, 1]]                  # (E,3)
        dot = (x_diff * self.edge_dirs).sum(dim=1, keepdim=True)  # (E,1)
        Hx = self.k * dot * self.edge_dirs                        # (E,3)
        accum = self.dt**2 * Hx                                   # (E,3)

        Ax = Mx
        Ax.index_add_(0, edges[:, 0], accum)
        Ax.index_add_(0, edges[:, 1], -accum)
        return Ax

    def rhs_b(self, pos: Tensor) -> Tensor:
        """RHS of system Ax = b in Wu et al. [2022, Section 3.1] used in PCG solver."""
        M = self.voxels.node_mass.unsqueeze(-1)  # (N, 1)
        v = self.voxels.node_vel
        force = self.internal_forces(pos) + self.external_forces(pos)
        b = (self.dt * M * v) + (self.dt**2 * force)
        return b

    def step(self, pos: Tensor, max_iters: int = 50, tol: float = 1e-5):
        """Computes a single implicit Euler step using a preconditioned congugate gradient solver."""
        N = pos.shape[0]
        self.refresh_edges(pos)

        def matvec(x_flat):
            x_t = torch.from_numpy(np.ascontiguousarray(x_flat)).reshape(N, 3)
            return self.lhs_Ax(x_t).reshape(-1).numpy()

        Ax = LinearOperator((3*N, 3*N), matvec=matvec, dtype=np.float32)
        b  = self.rhs_b(pos).reshape(-1).numpy()
        x_init = (self.dt * self.voxels.node_vel).reshape(-1).numpy()
        delta_pos, info = cg(Ax, b, x0=x_init, rtol=tol, maxiter=max_iters)

        if info > 0:
            # solver didn't converge
            pass

        delta_pos = torch.from_numpy(delta_pos).reshape(N, 3).clone()
        self.voxels.node_vel = delta_pos / self.dt
        return pos + delta_pos
