import numpy as np
import torch
from torch import Tensor
from scipy.sparse.linalg import LinearOperator, cg

from src.scene import Scene

def solve_step(scene: Scene, pos: Tensor, max_iters: int = 50, tol: float = 1e-5):
    """Computes a single implicit Euler step using a preconditioned congugate gradient solver."""
    scene.refresh_edges(pos)
    N = pos.shape[0]
    dtype = np.float32

    def matvec(x_flat):
        x_t = torch.frosm_numpy(np.ascontiguousarray(x_flat)).reshape(N, 3)
        return scene.lhs_Ax(x_t).reshape(-1).numpy()

    Ax = LinearOperator((3*N, 3*N), matvec=matvec, dtype=dtype)
    b  = scene.rhs_b(pos).reshape(-1).numpy()
    x_init = (scene.dt * scene.voxels.node_vel).reshape(-1).numpy()

    delta_pos, info = cg(Ax, b, x0=x_init, rtol=tol, maxiter=max_iters)

    if info > 0:
        # solver didn't converge
        pass

    delta_pos = torch.from_numpy(delta_pos).reshape(N, 3).clone()
    scene.voxels.node_vel = delta_pos / scene.dt
    return pos + delta_pos
