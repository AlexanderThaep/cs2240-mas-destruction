import torch
from torch import Tensor
from voxels import Voxels
from dataclasses import dataclass

# TODO: add a guard in a single location to ensure that there is >= 1 voxel.

@dataclass
class Scene:
    def __init__(
        self,
        voxels: Voxels,
        stiffness: float,
        dt: float,
        gravity = Tensor([0.0, -9.8, 0.0]),
    ):
        self.voxels = voxels
        self.k = stiffness
        self.dt = dt
        self.gravity = gravity
        self.edge_dirs = None  # this is updated as things break

    def refresh_edges(self, pos: Tensor):
        edges = self.voxels.edges
        edge_vecs = pos[edges[:, 1]] - pos[edges[:, 0]]
        edge_lens = edge_vecs.norm(dim=1).clamp(min=1e-10).unsqueeze(-1)
        self.edge_dirs = edge_vecs / edge_lens

    def external_forces(self, pos: Tensor) -> Tensor:
        """External forces felt by each node"""
        forces = torch.zeros_like(pos)  # (N,3)

        gravity = self.voxels.node_mass.unsqueeze(-1) * self.gravity.unsqueeze(0)

        """[other] forces here will have to encode other things
        such as collisions and explosive forces"""
        other = torch.zeros_like(pos)

        forces += gravity + other

        return forces

    def internal_forces(self, pos: Tensor) -> Tensor:
        """Internal spring forces felt by each node (sum over Hooke's Law spring forces)."""
        forces = torch.zeros_like(pos)  # (N,3)

        edges = self.voxels.edges                                         # (E,2)
        d = pos[edges[:, 1]] - pos[edges[:, 0]]                           # (E,3)
        L = d.norm(dim=1).clamp(min=1e-10)                                # (E)
        dhat = d / L.unsqueeze(-1)                                        # (E,3)
        fi = (self.k * (L - self.voxels.edge_rest)).unsqueeze(-1) * dhat  # (E,3)

        forces.index_add_(0, edges[:, 0],  fi)
        forces.index_add_(0, edges[:, 1], -fi)
        return forces

    def lhs_Ax(self, x: Tensor) -> Tensor:
        """LHS of system Ax = b in Wu et al. [2022, Section 3.1] used in PCG solver."""
        Mx = self.voxels.node_mass.unsqueeze(-1) * x # (N, 1))
        edges = self.voxels.edges                                 # (E,3)
        edge_vecs = edges[:, 1] - edges[: 0]                      # (E,3)
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
