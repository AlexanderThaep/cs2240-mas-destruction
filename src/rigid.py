import torch
from quatorch import Quaternion
from dataclasses import dataclass, field
from typing import Dict, Tuple

def apply_gravity(
    gravity,  # (3,)
    velocity, # (3,)
    dt       
):
    """
    Updates:
    - velocity
    """

    velocity = velocity + gravity * dt 

    return velocity

def integrate_rigid_body(
    mass,
    com,              # (3,) tensor
    velocity,         # (3,)
    angular_velocity, # (3,) ω
    orientation,      # Quaternion (quatorch)
    dt
):
    """
    Updates:
    - center of mass
    - orientation via quaternion integration
    """

    # --- translation ---
    com = com + velocity * dt

    # --- angular velocity quaternion (pure quaternion) ---
    omega_quat = Quaternion(
        torch.tensor([0.0,
                      angular_velocity[0],
                      angular_velocity[1],
                      angular_velocity[2]], dtype=com.dtype)
    )

    # --- dq/dt = 0.5 * ω ⊗ q ---
    dq = omega_quat * orientation * 0.5

    # --- integrate ---
    orientation = orientation + dq * dt
    orientation = orientation.normalized()

    return com, orientation

def apply_tra_voxels(
    world_coords: torch.Tensor,  # (V,3) world-space voxel positions
    old_com: torch.Tensor,       # (3,) old COM
    com: torch.Tensor            # (3,) new COM
):
    """
    Applies translation to world-space voxels based on COM delta.
    """

    pts = world_coords.to(dtype=com.dtype)
    delta = com - old_com

    return pts + delta

def apply_rot_voxels(
    world_coords: torch.Tensor,  # (V,3) world-space voxel positions
    com: torch.Tensor,           # (3,)
    orientation: Quaternion      # unit quaternion
):
    """
    Applies rotation about COM to world-space voxels.
    """

    pts = world_coords.to(dtype=com.dtype)
    rel = pts - com
    rotated = orientation.rotate_vector(rel)
    world_coords = rotated + com

    return world_coords