import torch
from torch import Tensor
from voxels import Voxels
from dataclasses import dataclass

@dataclass
class Scene:
    def __init__(
        self,
        voxels: Voxels,
        gravity = Tensor(0.0, -9.8, 0.0)
    ):
        self.voxels = voxels

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

        if self.voxels.node_rest is None:
            return forces
        
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
        pass

    def rhs_b(self, pos: Tensor) -> Tensor:
        """RHS of system Ax = b in Wu et al. [2022, Section 3.1] used in PCG solver."""
        pass
    
        