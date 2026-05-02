import torch
from torch import Tensor
from voxels import Voxels
from dataclasses import dataclass
from scipy.sparse.linalg import LinearOperator, cg
import numpy as np

# TODO: add a guard in a single location to ensure that there is >= 1 voxel.

@dataclass
class Scene:
    voxels:     Voxels                             # voxels in scene
    k:          float                              # spring constant (stiffness)
    dt:         float                              # Euler approximation timestep
    gravity:    Tensor = Tensor([0.0, -9.8, 0.0])  # downward gravitational acceleration
    edge_lens:  Tensor = None                      # edge lengths
    edge_dirs:  Tensor = None                      # edge directions (updated as things break)

    def refresh_edges(self, pos: Tensor):
        edges = self.voxels.edges
        edge_vecs = pos[edges[:, 1]] - pos[edges[:, 0]]
        edge_lens = edge_vecs.norm(dim=1).clamp(min=1e-10)
        self.edge_lens = edge_lens
        self.edge_dirs = edge_vecs / edge_lens.unsqueeze(-1)

    def external_forces(self, pos: Tensor) -> Tensor:
        """External forces felt by each node"""
        forces = torch.zeros_like(pos)  # (N,3)
        gravity = self.voxels.node_mass.unsqueeze(-1) * self.gravity.unsqueeze(0)
        return forces + gravity

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
