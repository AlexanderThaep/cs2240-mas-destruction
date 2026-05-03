import torch
from torch import Tensor
from voxels import Voxels
from morton import morton_code
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.gravity = self.gravity.to(device)
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
        offsets = torch.cartesian_prod(*[torch.arange(-1, 2)] * 3).to(device)                     # (27,3)
        neighbor_keys = morton_code(cells.unsqueeze(1) + offsets.unsqueeze(0) - origin)  # (V,27)
        low = torch.searchsorted(keys, neighbor_keys.reshape(-1), right=False)           # (27V)
        high = torch.searchsorted(keys, neighbor_keys.reshape(-1), right=True)           # (27V)
        count = high - low                                                               # (27V)
        total = int(count.sum())

        if total == 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty

        query_voxel = torch.arange(V).repeat_interleave(27).to(device)  # (27V)
        a = query_voxel.repeat_interleave(count)                        # (total)
        range_start = low.repeat_interleave(count)                      # (total)
        query_offset = torch.cumsum(count, dim=0) - count
        pair_offset = query_offset.repeat_interleave(count).to(device)  # (total)
        offset = torch.arange(total).to(device) - pair_offset
        b = sort_idx[range_start + offset]                              # (total)

        keep = a < b
        return a[keep], b[keep]

    def project_ground(self, pos: Tensor) -> Tensor:
        """Clamp nodes above the ground plane."""
        new_pos = pos.clone()
        new_pos[:, 1] = pos[:, 1].clamp(min=self.ground_y)
        return new_pos

    def project_collisions(self, pos: Tensor, iters: int = 4) -> Tensor:
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

        K = a.shape[0]

        # Candidate voxel rotations
        unique_v, inv = torch.unique(torch.cat([a, b]), return_inverse=True)
        R_unique = self.voxels.voxel_rotations(pos, indices=unique_v)
        R_a = R_unique[inv[:K]]                                              # (K,3,3)
        R_b = R_unique[inv[K:]]
        a_axes = R_a.transpose(-1, -2)                                       # (K,3,3)
        b_axes = R_b.transpose(-1, -2)

        # Half-extents along local axes (computed from initial corners)
        a_corners0 = pos[nodes[a]]                                           # (K,8,3)
        b_corners0 = pos[nodes[b]]
        a_local = (a_corners0 - a_corners0.mean(dim=1, keepdim=True)) @ R_a
        b_local = (b_corners0 - b_corners0.mean(dim=1, keepdim=True)) @ R_b
        a_ext = a_local.abs().max(dim=1).values                              # (K,3)
        b_ext = b_local.abs().max(dim=1).values

        # 15 SAT axes (6 face normals + 9 edge cross products)
        face_axes = torch.cat([a_axes, b_axes], dim=1)                       # (K,6,3)
        edge_axes = torch.linalg.cross(
            a_axes.unsqueeze(2).expand(-1, -1, 3, -1),                       # (K,3,3,3)
            b_axes.unsqueeze(1).expand(-1, 3, -1, -1),
            dim=-1,
        ).reshape(K, 9, 3)
        all_axes = torch.cat([face_axes, edge_axes], dim=1)                  # (K,15,3)
        norms = all_axes.norm(dim=-1, keepdim=True)
        degenerate = norms.squeeze(-1) < 1e-6                                # (K,15)
        all_axes = all_axes / norms.clamp(min=1e-10)

        # Per-OBB radius along each test axis
        a_dots = torch.einsum('kij,klj->kil', a_axes, all_axes)              # (K,3,15)
        b_dots = torch.einsum('kij,klj->kil', b_axes, all_axes)
        r_sum = (a_ext.unsqueeze(-1) * a_dots.abs()).sum(dim=1) \
              + (b_ext.unsqueeze(-1) * b_dots.abs()).sum(dim=1)              # (K,15)

        # Bias (small penalty on edge-cross axes so face axes win on ties)
        bias = torch.zeros(15, dtype=pos.dtype, device=pos.device)
        bias[6:] = h * 1e-3

        new_pos = pos
        idx = torch.arange(K)
        INF = float('inf')

        for _ in range(iters):
            a_c = new_pos[nodes[a]].mean(dim=1)                                        # (K,3)
            b_c = new_pos[nodes[b]].mean(dim=1)
            cdiff = b_c - a_c                                                          # (K,3)
            d = (cdiff.unsqueeze(1) * all_axes).sum(dim=-1).abs()                      # (K,15)
            overlap = r_sum - d                                                        # (K,15)
            overlap = torch.where(degenerate, torch.full_like(overlap, INF), overlap)

            min_idx = (overlap + bias).argmin(dim=1)                                   # (K)
            depth = overlap.gather(1, min_idx.unsqueeze(-1)).squeeze(-1)               # (K)
            contact = depth > 0

            if not contact.any():
                break

            ai, bi = a[contact], b[contact]
            pop = all_axes[idx, min_idx][contact]                                      # (Kc,3)
            cdc = cdiff[contact]
            dpc = depth[contact]
            sgn = (cdc * pop).sum(dim=-1).sign()
            sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
            push = (0.5 * dpc * sgn).unsqueeze(-1) * pop                               # (Kc,3)

            Kc = ai.shape[0]
            push_corners = push.repeat_interleave(8, dim=0)
            ones_corners = torch.ones(Kc * 8).to(device)
            correction = torch.zeros_like(new_pos).to(device)
            count = torch.zeros(new_pos.shape[0]).to(device)
            correction.index_add_(0, nodes[bi].reshape(-1),  push_corners)
            correction.index_add_(0, nodes[ai].reshape(-1), -push_corners)
            count.index_add_(0, nodes[bi].reshape(-1), ones_corners)
            count.index_add_(0, nodes[ai].reshape(-1), ones_corners)
            new_pos = new_pos + correction / count.clamp(min=1).unsqueeze(-1)

        return new_pos

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
        va_all = va_all.to(device)
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

    def diagonal(self) -> torch.Tensor:
        # mass part
        diag = self.voxels.node_mass.unsqueeze(-1).expand(-1, 3).clone()
        e = self.voxels.edges
        if e.shape[0] == 0:
            return diag
        # diag of k dhat dhat^T is k * dhat^2
        contrib = (self.dt * self.dt * self.k) * (self.edge_dirs * self.edge_dirs)  # (E,3)
        diag.index_add_(0, e[:, 0], contrib)
        diag.index_add_(0, e[:, 1], contrib)
        return 1 / diag

    def _pcg(self, b: Tensor, x0: Tensor, max_iters: int, tol: float) -> Tensor:
        b_norm_sq = (b * b).sum()
        if b_norm_sq < 1e-30:
            return torch.zeros_like(x0)
        b_norm = b_norm_sq.sqrt()

        x = x0.clone()
        r = b - self.lhs_Ax(x)
        z = r * self.diag
        p = z.clone()
        rz = (r * z).sum()
        for _ in range(max_iters):
            Ap = self.lhs_Ax(p)
            alpha = rz / (p * Ap).sum().clamp(min=1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            if (r.norm() / b_norm) < tol:
                break
            z = r * self.diag
            rz_new = (r * z).sum()
            beta = rz_new / rz
            p = z + beta * p
            rz = rz_new
        return x

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

    def step(self, max_iters: int = 10000, tol: float = 1e-5) -> int:
        """Computes a single implicit Euler step using a preconditioned congugate gradient solver."""
        pos = self.voxels.node_pos
        self.refresh_edges(pos)
        b  = self.rhs_b(pos)
        x0 = self.dt * self.voxels.node_vel
        self.diag = self.diagonal()
        new_pos = pos + self._pcg(b, x0, max_iters, tol)
        if self.self_collide:
            new_pos = self.project_collisions(new_pos)
        new_pos = self.project_ground(new_pos)
        self.voxels.node_vel = (new_pos - pos) / self.dt
        self.voxels.node_pos = new_pos
        return self.fracture()
