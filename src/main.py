import torch

from mesh import Mesh
from voxels import Voxels
from simulation import Simulation
import window
import acceleration

acceleration.device_info()
device = acceleration.get_device()

# Two knights charging at each other.
left  = Mesh.from_vox("objects/character/chr_knight.vox")
right = Mesh.from_vox("objects/character/chr_knight.vox").translate([40, 0, 0])

voxels, (left_nodes, right_nodes) = Voxels.from_meshes([left, right], h=1.0)
voxels.init_state(density=1.0)

voxels.node_vel[left_nodes]  = torch.tensor([ 50.0, 0.0, 0.0]).to(device)
voxels.node_vel[right_nodes] = torch.tensor([-50.0, 0.0, 0.0]).to(device)

sim = Simulation(
    voxels        = voxels,
    k             = 1e5,
    dt            = 1/3600,
    ground_y      = 0.0,
    self_collide  = True,
    do_fracture   = True,
    tensile_yield = 0.01,
)

print(f"V={voxels.V}  N={voxels.N}  E={voxels.E}")
window.run(sim, title="knight vs knight")

# Target scene
# castle     = Mesh.from_vox("objects/monument/monu7.vox")
# projectile = Mesh.from_vox("objects/monument/monu5.vox").translate([200, 60, 0])
# voxels, (castle_nodes, proj_nodes) = Voxels.from_meshes([castle, projectile], h=1.0)
# voxels.init_state(density=1.0)
# voxels.node_vel[proj_nodes] = torch.tensor([-100.0, 0.0, 0.0]).to(device)
# sim = Simulation(voxels=voxels, k=1e3, dt=1/60, ground_y=0.0,
#                  self_collide=True, do_fracture=True, tensile_yield=0.05)
# window.run(sim, title="monu5 -> monu7")
