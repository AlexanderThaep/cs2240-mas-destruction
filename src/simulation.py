import torch
from torch import Tensor
from voxels import Voxels
from morton import morton_code
from dataclasses import dataclass

import acceleration
from jacobi import JacobiPreconditioner

device = acceleration.get_device()

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
    do_fracture:     bool  = True   # breaking links between nodes
    compliance:      float = 1e-6   # XPBD contact compliance (0 = hard PBD; bigger = softer)
    friction:        float = 1.0    # Coulomb friction coefficient

    # fracture
    tensile_yield:  float = 0.15  # break links beyond this stretch amount

    # caches
    edge_lens:  Tensor = None
    edge_dirs:  Tensor = None

    # preconditioner (for pcg)
    precond: object = None

    def __post_init__(self):
        self.gravity = self.gravity.to(device)
        assert self.voxels.V >= 1, "Simulation requires at least one voxel"
        assert self.voxels.node_pos is not None, "Call voxels.init_state() before constructing Simulation"
        if self.precond is None:
            self.precond = JacobiPreconditioner(self)

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
        sorted, sort_idx = torch.sort(morton_codes) # (V), voxels on Z-order curve

        # In 3D each cell is in a 3x3x3 = 27 cell neighborhood
        offsets = torch.cartesian_prod(*[torch.arange(-1, 2)] * 3).to(device)            # (27,3)
        neighbor_keys = morton_code(cells.unsqueeze(1) + offsets.unsqueeze(0) - origin)  # (V,27)
        low = torch.searchsorted(sorted, neighbor_keys.reshape(-1), right=False)         # (27V)
        high = torch.searchsorted(sorted, neighbor_keys.reshape(-1), right=True)         # (27V)
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

    def project_ground(self, pos: Tensor, prev_pos: Tensor = None) -> Tensor:
        """Clamp nodes above ground plane; apply Coulomb tangential friction if prev_pos given."""
        new_pos = pos.clone()
        pen = (self.ground_y - pos[:, 1]).clamp(min=0)  # (N)
        new_pos[:, 1] = pos[:, 1] + pen

        if self.friction > 0 and prev_pos is not None:
            ic = pen > 0
            if ic.any():
                xz = pos[ic][:, [0, 2]]
                xz0 = prev_pos[ic][:, [0, 2]]
                slide = xz - xz0                                         # (Kc, 2)
                mag = slide.norm(dim=-1, keepdim=True).clamp(min=1e-10)
                cap = (self.friction * pen[ic]).unsqueeze(-1)
                scale = (cap / mag).clamp(max=1.0)
                new_xz = xz - slide * scale
                new_pos[ic, 0] = new_xz[:, 0]
                new_pos[ic, 2] = new_xz[:, 1]
        return new_pos

    @staticmethod
    def _obb_half_extents(corners: Tensor, R: Tensor) -> Tensor:
        """Half-extents of OBBs given their (K,8,3) corners and (K,3,3) rotations."""
        centered = corners - corners.mean(dim=1, keepdim=True)
        return (centered @ R).abs().max(dim=1).values  # (K,3)

    def project_collisions(self, pos: Tensor, prev_pos: Tensor = None, iters: int = 4) -> Tensor:
        """Push apart overlapping voxels via XPBD with self.compliance and self.friction."""
        h = self.voxels.h
        nodes = self.voxels.voxel_nodes
        a, b = self.collision_candidates(pos)
        if a.numel() == 0:
            return pos

        # Drop pairs that already share a node
        shares = (nodes[a].unsqueeze(-1) == nodes[b].unsqueeze(-2)).any(-1).any(-1)
        a, b = a[~shares], b[~shares]
        if a.numel() == 0:
            return pos

        K = a.shape[0]
        nodes_a, nodes_b = nodes[a], nodes[b]                                # (K,8)

        # Per-pair OBB rotations and half-extents
        unique_v, inv = torch.unique(torch.cat([a, b]), return_inverse=True)
        R_uniq = self.voxels.voxel_rotations(pos, indices=unique_v)
        R_a, R_b = R_uniq[inv[:K]], R_uniq[inv[K:]]                          # (K,3,3)
        Aa, Ab = R_a.transpose(-1, -2), R_b.transpose(-1, -2)                # rows = local axes
        ext_a = self._obb_half_extents(pos[nodes_a], R_a)                    # (K,3)
        ext_b = self._obb_half_extents(pos[nodes_b], R_b)

        # 15 SAT axes: 6 face normals + 9 edge crosses
        edge_ax = torch.linalg.cross(
            Aa.unsqueeze(2).expand(-1, -1, 3, -1),
            Ab.unsqueeze(1).expand(-1, 3, -1, -1),
            dim=-1,
        ).reshape(K, 9, 3)
        axes = torch.cat([Aa, Ab, edge_ax], dim=1)                           # (K,15,3)
        norms = axes.norm(dim=-1, keepdim=True)
        degenerate = norms.squeeze(-1) < 1e-6                                # (K,15)
        axes = axes / norms.clamp(min=1e-10)

        # Per-OBB radius along each test axis
        a_proj = (axes @ Aa.transpose(-1, -2)).abs() * ext_a.unsqueeze(1)    # (K,15,3)
        b_proj = (axes @ Ab.transpose(-1, -2)).abs() * ext_b.unsqueeze(1)
        r_sum = a_proj.sum(-1) + b_proj.sum(-1)                              # (K,15)

        # Bias (prefer face axes over edge crosses on ties)
        bias = torch.zeros(15, dtype=pos.dtype, device=device)
        bias[6:] = h * 1e-3

        # XPBD (per-pair Lagrange multiplier accumulated across iterations)
        w = self.voxels.V / self.voxels.node_mass.sum()
        alpha = self.compliance / (self.dt * self.dt)
        lam = torch.zeros(K, device=device)

        correction = torch.zeros_like(pos)
        count      = torch.zeros(pos.shape[0], device=device)
        idx        = torch.arange(K, device=device)
        INF        = float('inf')
        new_pos    = pos

        for _ in range(iters):
            cdiff = new_pos[nodes_b].mean(1) - new_pos[nodes_a].mean(1)              # (K,3)
            d = (cdiff.unsqueeze(1) * axes).sum(-1).abs()                            # (K,15)
            overlap = torch.where(degenerate, torch.full_like(d, INF), r_sum - d)
            min_idx = (overlap + bias).argmin(dim=1)                                 # (K)
            depth = overlap.gather(1, min_idx.unsqueeze(-1)).squeeze(-1)             # (K) signed

            # XPBD multiplier update
            d_lam = (depth - alpha * lam) / (2 * w + alpha)
            lam_new = (lam + d_lam).clamp(min=0)
            eff = lam_new - lam
            lam = lam_new

            active = eff != 0
            if not active.any():
                break

            pop = axes[idx, min_idx][active]                                         # (Ka,3)
            s = (cdiff[active] * pop).sum(-1)
            sign = torch.sign(s) + (s == 0).to(s.dtype)
            push = (w * eff[active] * sign).unsqueeze(-1) * pop                      # (Ka,3)
            push8 = push.repeat_interleave(8, dim=0)
            ones8 = torch.ones(push8.shape[0], device=device)
            na, nb = nodes_a[active].reshape(-1), nodes_b[active].reshape(-1)

            correction.zero_(); count.zero_()
            correction.index_add_(0, nb,  push8); correction.index_add_(0, na, -push8)
            count.index_add_(0, nb, ones8);       count.index_add_(0, na, ones8)
            new_pos = new_pos + correction / count.clamp(min=1).unsqueeze(-1)

        # Coulomb friction
        if self.friction > 0 and prev_pos is not None and (lam > 0).any():
            ic = lam > 0
            cdiff = new_pos[nodes_b].mean(1) - new_pos[nodes_a].mean(1)
            d = (cdiff.unsqueeze(1) * axes).sum(-1).abs()
            overlap = torch.where(degenerate, torch.full_like(d, INF), r_sum - d)
            min_idx = (overlap + bias).argmin(dim=1)
            pop = axes[idx, min_idx][ic]
            s = (cdiff[ic] * pop).sum(-1)
            sign = torch.sign(s) + (s == 0).to(s.dtype)
            n_hat = sign.unsqueeze(-1) * pop                                     # (Kc,3)

            ca, cb  = new_pos[nodes_a[ic]].mean(1),  new_pos[nodes_b[ic]].mean(1)
            ca0, cb0 = prev_pos[nodes_a[ic]].mean(1), prev_pos[nodes_b[ic]].mean(1)
            rel_t = (cb - cb0) - (ca - ca0)
            rel_t = rel_t - (rel_t * n_hat).sum(-1, keepdim=True) * n_hat
            rel_t_mag = rel_t.norm(dim=-1, keepdim=True).clamp(min=1e-10)

            max_t = (self.friction * 2 * w * lam[ic]).unsqueeze(-1)
            scale = (max_t / rel_t_mag).clamp(max=1.0)
            push = -0.5 * rel_t * scale

            push8 = push.repeat_interleave(8, dim=0)
            ones8 = torch.ones(push8.shape[0], device=device)
            na, nb = nodes_a[ic].reshape(-1), nodes_b[ic].reshape(-1)

            correction.zero_()
            count.zero_()

            correction.index_add_(0, nb,  push8)
            correction.index_add_(0, na, -push8)
            count.index_add_(0, nb, ones8)
            count.index_add_(0, na, ones8)

            new_pos = new_pos + correction / count.clamp(min=1).unsqueeze(-1)

        return new_pos

    def external_forces(self, pos: Tensor) -> Tensor:
        """External forces felt by each node."""
        return self.voxels.node_mass.unsqueeze(-1) * self.gravity.unsqueeze(0)

    def internal_forces(self, pos: Tensor) -> Tensor:
        """Internal spring forces felt by each node (sum over Hooke's Law spring forces)."""
        forces = torch.zeros_like(pos)                      # (N,3)
        edges = self.voxels.edges                           # (E,2)
        L, L0 = self.edge_lens, self.voxels.edge_lens_rest  # (E)
        mult = self.voxels.edge_mult                        # (E) per-edge stiffness scale

        force_i = (self.k * mult * (L - L0)).unsqueeze(-1) * self.edge_dirs  # (E,3)

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
        Mx = self.voxels.node_mass.unsqueeze(-1) * x               # (N,1)
        edges = self.voxels.edges                                  # (E,2)
        mult = self.voxels.edge_mult                               # (E)
        x_diff = x[edges[:, 0]] - x[edges[:, 1]]                   # (E,3)
        dot = (x_diff * self.edge_dirs).sum(dim=1, keepdim=True)   # (E,1)
        Hx = (self.k * mult).unsqueeze(-1) * dot * self.edge_dirs  # (E,3)
        accum = self.dt**2 * Hx                                    # (E,3)

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

    def _pcg(self, b: Tensor, x0: Tensor, max_iters: int, tol: float) -> Tensor:
        b_norm_sq = (b * b).sum()
        if b_norm_sq < 1e-30:
            return torch.zeros_like(x0)
        b_norm = b_norm_sq.sqrt()

        x = x0.clone()
        r = b - self.lhs_Ax(x)
        z = self.precond.apply(r)
        p = z.clone()
        rz = (r * z).sum()
        for _ in range(max_iters):
            Ap = self.lhs_Ax(p)
            alpha = rz / (p * Ap).sum().clamp(min=1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            if (r.norm() / b_norm) < tol:
                break
            z = self.precond.apply(r)
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

    def step(self, max_iters: int = 100000, tol: float = 1e-5) -> int:
        """Computes a single implicit Euler step using a preconditioned congugate gradient solver."""
        pos = self.voxels.node_pos
        self.refresh_edges(pos)
        b  = self.rhs_b(pos)
        x0 = self.dt * self.voxels.node_vel
        self.precond.rebuild()
        new_pos = pos + self._pcg(b, x0, max_iters, tol)
        if self.self_collide:
            new_pos = self.project_collisions(new_pos, prev_pos=pos)
        new_pos = self.project_ground(new_pos, prev_pos=pos)
        self.voxels.node_vel = (new_pos - pos) / self.dt
        self.voxels.node_pos = new_pos
        if self.do_fracture:
            return self.fracture()
        else:
            return 0
