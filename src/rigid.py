import torch
from quatorch import Quaternion
from dataclasses import dataclass, field
from typing import Dict, Tuple

Q_IDENTITY = Quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))

def compute_voxel_com(
    world_coords
):
    pts = world_coords.to(dtype=torch.float32)
    return pts.mean(dim=0)

def compute_voxel_inertia(
    com,
    world_coords, 
    mass
):
    pts = world_coords - com
    V = pts.shape[0]
    m = mass / V

    r2 = (pts * pts).sum(dim=1)

    term1 = m * torch.sum(r2) * torch.eye(3, dtype=pts.dtype, device=pts.device)
    term2 = m * (pts.T @ pts)

    I = term1 - term2
    return I

def integrate_rigid_body(
    mass,
    com,              # (3,) tensor
    velocity,         # (3,)
    angular_velocity, # (3,) ω
    orientation,      # Quaternion (quatorch)
    inertia_body,     # (3,3) inertia in body frame
    torque,           # (3,) world torque
    force,            # (3,) world force
    dt
):
    """
    Updates:
    - center of mass
    - velocity
    - orientation via quaternion integration
    - angular velocity
    """

    acceleration = force / mass
    velocity = velocity + acceleration * dt
    com = com + velocity * dt

    R = orientation.rotation_matrix()
    I_world = R @ inertia_body @ R.T
    I_inv = torch.inverse(I_world)

    omega = angular_velocity
    omega_dot = I_inv @ (torque - torch.cross(omega, I_world @ omega))
    angular_velocity = angular_velocity + omega_dot * dt

    omega_quat = Quaternion(torch.tensor([
        0.0,
        angular_velocity[0],
        angular_velocity[1],
        angular_velocity[2]
    ], dtype=com.dtype))

    dq = 0.5 * omega_quat * orientation
    orientation = orientation + dq * dt
    orientation = orientation.normalized()

    return com, velocity, orientation, angular_velocity

def apply_voxels(
    world_coords: torch.Tensor,  # (V,3) world-space voxel positions
    old_com: torch.Tensor,       # (3,) old COM
    com: torch.Tensor,           # (3,) new COM
    orientation=Q_IDENTITY       # unit quaternion
):
    """
    Applies translation to world-space voxels based on COM delta
    and applies rotation about COM to world-space voxels.
    """

    pts = world_coords.to(dtype=com.dtype)
    delta = com - old_com
    rel = pts - com
    rotated = orientation.rotate_vector(rel)

    return rotated + (pts + delta)