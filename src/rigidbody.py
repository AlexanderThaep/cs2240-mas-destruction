import torch
from quatorch import Quaternion
from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class RigidBody:
    # constants (only change upon fracture)
    mass:          float
    inertia_body:  torch.Tensor   # (3,3) in body frame
    # body_offsets:  torch.Tensor   # (V_b, 3) voxel centers in body frame
    # voxel_ids:     torch.Tensor   # (V_b,)   which mesh voxels belong to this body

    # simulation state
    com:           torch.Tensor   # (3,)
    orientation:   Quaternion
    velocity:      torch.Tensor   # (3,)
    angular_vel:   torch.Tensor   # (3,) world frame

    def _init_com(self, world_coords):
        pts = world_coords.to(dtype=torch.float32)
        self.com = pts.mean(dim=0)

    def _init_voxel_inertia(self, voxel_coords):
        pts = (voxel_coords + 0.5) - self.com

        V = pts.shape[0]
        m = self.mass / V

        r2 = (pts * pts).sum(dim=1)

        term1 = m * torch.sum(r2) * torch.eye(3, dtype=pts.dtype, device=pts.device)
        term2 = m * (pts.T @ pts)

        I = term1 - term2
        self.inertia_body = I

    def integrate_rigid_body(
        self,
        torque,           # (3,) world torque
        force,            # (3,) world force
        dt
    ):
        acceleration = force / self.mass
        self.velocity += acceleration * dt
        self.com += self.velocity * dt

        R = self.orientation.to_rotation_matrix()
        I_world = R @ self.inertia_body @ R.T
        I_inv = torch.inverse(I_world)

        omega = self.angular_vel
        omega_dot = I_inv @ (torque - torch.cross(omega, I_world @ omega))
        angular_velocity = self.angular_vel + omega_dot * dt

        omega_quat = Quaternion(torch.tensor([
            0.0,
            angular_velocity[0],
            angular_velocity[1],
            angular_velocity[2]
        ], dtype=self.com.dtype))

        dq = 0.5 * omega_quat * self.orientation
        o = self.orientation + dq * dt
        orientation = o.normalize()

        self.orientation = orientation

    def apply_voxels(self, world_coords):
        pts = world_coords.to(dtype=self.com.dtype)
        rel = pts - self.com
        rotated = self.orientation.rotate_vector(rel)
        return rotated + self.com