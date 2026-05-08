from torch import Tensor
import acceleration

device = acceleration.get_device()


class JacobiPreconditioner:
    """Diagonal preconditioner M^{-1} = diag(A)^{-1}."""
    def __init__(self, sim):
        self.sim = sim
        self.diag_inv: Tensor = None

    def rebuild(self) -> None:
        sim, v = self.sim, self.sim.voxels
        diag = v.node_mass.unsqueeze(-1).expand(-1, 3).clone()
        if v.edges.shape[0]:
            contrib = (sim.dt * sim.dt * sim.k) * (sim.edge_dirs * sim.edge_dirs)
            diag.index_add_(0, v.edges[:, 0], contrib)
            diag.index_add_(0, v.edges[:, 1], contrib)
        self.diag_inv = 1.0 / diag

    def apply(self, r: Tensor) -> Tensor:
        return r * self.diag_inv
