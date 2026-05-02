import torch
from torch import Tensor
from voxels import Voxels
from morton import morton_code
from dataclasses import dataclass

@dataclass
class Simulation:
    # simulation parameters
    voxels:   Voxels                             # voxels in sim
    k:        float                              # spring constant (stiffness)
    dt:       float                              # Euler approximation timestep
    gravity:  Tensor = Tensor([0.0, -9.8, 0.0])  # downward gravitational acceleration

    # collisions (resolved by position projection, not penalty forces)
    ground_y:        float = 0.0    # height of ground plane
    self_collide:    bool  = True   # voxel-voxel inter-component collisions

    # fracture
    tensile_yield:  float = 0.15  # break links beyond this stretch amount

    # caches
    edge_lens:  Tensor = None
    edge_dirs:  Tensor = None

    def __post_init__(self):
        assert self.voxels.V >= 1, "Simulation requires at least one voxel"
        assert self.voxels.node_pos is not None, "Call voxels.init_state() before constructing Simulation"

    def refresh_edges(self, pos: Tensor):
        edges = self.voxels.edges
        edge_vecs = pos[edges[:, 1]] - pos[edges[:, 0]]
        edge_lens = edge_vecs.norm(dim=1).clamp(min=1e-10)
        self.edge_lens = edge_lens
        self.edge_dirs = edge_vecs / edge_lens.unsqueeze(-1)

    def collision_candidates(self, pos: Tensor) -> tuple[Tensor, Tensor]:
        """Returns candidate voxel pairs that could be colliding based on spatial locality."""
        h = self.voxels.h
        nodes = self.voxels.voxel_nodes         # (V,8)
        V = nodes.shape[0]

        centroids = pos[nodes].mean(dim=1)          # (V,3)
        cells = (centroids / h).floor().long()      # (V,3)
        origin = cells.min(dim=0).values - 1        # (3)
        morton_codes = morton_code(cells - origin)  # (V)
        sort_idx = morton_codes.argsort()           # (V)
        keys = morton_codes[sort_idx]               # voxels on Z-order curve

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

        query_voxel = torch.arange(V).repeat_interleave(27)  # (27V)
        a = query_voxel.repeat_interleave(count)             # (total)
        range_start = low.repeat_interleave(count)           # (total)
        query_offset = torch.cumsum(count, dim=0) - count
        pair_offset = query_offset.repeat_interleave(count)  # (total)
        offset = torch.arange(total) - pair_offset
        b = sort_idx[range_start + offset]                   # (total)

        keep = a < b
        return a[keep], b[keep]

    def project_ground(self, pos: Tensor) -> Tensor:
        """Clamp nodes above the ground plane."""
        new_pos = pos.clone()
        new_pos[:, 1] = pos[:, 1].clamp(min=self.ground_y)
        return new_pos

    def project_collisions(self, pos: Tensor) -> Tensor:
        """Push apart overlapping voxels from different components (PBD-style)."""
        h = self.voxels.h
        nodes = self.voxels.voxel_nodes
        a, b = self.collision_candidates(pos)
        if a.numel() == 0:
            return pos

        components = self.voxels.connected_components()
        cross = components[a] != components[b]
        a, b = a[cross], b[cross]
        if a.numel() == 0:
            return pos

        centroids = pos[nodes].mean(dim=1)
        diff = centroids[b] - centroids[a]
        dist = diff.norm(dim=-1).clamp(min=1e-10)
        overlap = (h - dist).clamp(min=0.0)
        contact = overlap > 0
        if not contact.any():
            return pos

        a, b, overlap, dist, diff = a[contact], b[contact], overlap[contact], dist[contact], diff[contact]
        push = (overlap / 2).unsqueeze(-1) * (diff / dist.unsqueeze(-1))   # (K,3) a -> b/2

        correction = torch.zeros_like(pos)
        count      = torch.zeros(pos.shape[0])
        K = a.shape[0]
        push_corners = push.repeat_interleave(8, dim=0)
        ones_corners = torch.ones(K * 8)
        correction.index_add_(0, nodes[b].reshape(-1),  push_corners)
        correction.index_add_(0, nodes[a].reshape(-1), -push_corners)
        count.index_add_(0, nodes[b].reshape(-1), ones_corners)
        count.index_add_(0, nodes[a].reshape(-1), ones_corners)
        return pos + correction / count.clamp(min=1).unsqueeze(-1)

    def external_forces(self, pos: Tensor) -> Tensor:
        """External forces felt by each node (gravity only; contacts are projected)."""
        return self.voxels.node_mass.unsqueeze(-1) * self.gravity.unsqueeze(0)

    def internal_forces(self, pos: Tensor) -> Tensor:
        """Internal spring forces felt by each node (sum over Hooke's Law spring forces)."""
        forces = torch.zeros_like(pos)  # (N,3)
        edges = self.voxels.edges                                     # (E,2)
        L, L0 = self.edge_lens, self.voxels.edge_lens_rest            # (E)

        force_i = (self.k * (L - L0)).unsqueeze(-1) * self.edge_dirs  # (E,3)

        forces.index_add_(0, edges[:, 0],  force_i)
        forces.index_add_(0, edges[:, 1], -force_i)
        return forces

    def fracture(self) -> int:
        """Break face links when stretch exceeds tensile yield (rebuilds voxels)."""
        pos = self.voxels.node_pos
        all_links = self.voxels.voxel_links  # (V,6)

        va_all = torch.arange(self.voxels.V).unsqueeze(-1).expand(-1, 6).reshape(-1)  # (6V)
        vb_all = all_links.reshape(-1)                                                # (6V)
        valid  = (vb_all >= 0) & (va_all < vb_all)
        links  = torch.stack([va_all[valid], vb_all[valid]], dim=-1)                  # (L,2)

        if links.numel() == 0:
            return 0

        nodes = self.voxels.voxel_nodes                                        # (V,8)
        centroids = pos[nodes].mean(dim=1)                                     # (V,3)
        dist = (centroids[links[:, 1]] - centroids[links[:, 0]]).norm(dim=-1)  # (L)
        stretch = (dist - self.voxels.h) / self.voxels.h                       # (L)
        broken = stretch > self.tensile_yield
        num_broken = int(broken.sum())

        if num_broken == 0:
            return 0

        self.voxels.break_links(links[broken])
        self.voxels.rebuild_after_fracture()
        self.edge_lens = None  # invalidate caches; sized to old voxels.E
        self.edge_dirs = None
        return num_broken

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

    def _cg(self, b: Tensor, x0: Tensor, max_iters: int, tol: float) -> Tensor:
        """Conjugate gradient solve x ~= A^-1 b for the SPD operator self.lhs_Ax."""
        b_norm_sq = (b * b).sum()
        if b_norm_sq < 1e-30:
            return torch.zeros_like(x0)  # A is SPD, so b=0 implies x=0 exactly
        b_norm = b_norm_sq.sqrt()

        x = x0.clone()
        r = b - self.lhs_Ax(x)
        p = r.clone()
        rs = (r * r).sum()
        for _ in range(max_iters):
            Ap = self.lhs_Ax(p)
            alpha = rs / (p * Ap).sum().clamp(min=1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = (r * r).sum()
            if (rs_new.sqrt() / b_norm) < tol:
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
        return x

    def step(self, max_iters: int = 50, tol: float = 1e-5) -> int:
        """Computes a single implicit Euler step using a preconditioned congugate gradient solver."""
        pos = self.voxels.node_pos
        self.refresh_edges(pos)
        b  = self.rhs_b(pos)
        x0 = self.dt * self.voxels.node_vel
        new_pos = pos + self._cg(b, x0, max_iters, tol)
        if self.self_collide:
            new_pos = self.project_collisions(new_pos)
        new_pos = self.project_ground(new_pos)
        self.voxels.node_vel = (new_pos - pos) / self.dt
        self.voxels.node_pos = new_pos
        return self.fracture()
