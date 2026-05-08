import sys
import json
from pathlib import Path
import torch

from mesh import Mesh
from voxels import Voxels
from simulation import Simulation
import acceleration
import window

def load_tensor(x, device=None):
    return torch.tensor(x, dtype=torch.float32, device=device)

def load_mesh(path):
    ext = Path(path).suffix

    if ext == ".py":
        return Mesh.from_py(path)

    if ext == ".vox":
        return Mesh.from_vox(path)

    raise ValueError(f"unsupported mesh format: {ext}")

def build_mesh(obj):
    mesh = load_mesh(obj["mesh"])

    if "scale" in obj:
        mesh = mesh.scale(obj["scale"])

    if "translate" in obj:
        mesh = mesh.translate(load_tensor(obj["translate"]))

    return mesh

def load_scene(path):
    with open(path, "r") as f:
        scene = json.load(f)

    acceleration.device_info()
    device = acceleration.get_device()

    objects = scene["objects"]

    meshes = []
    velocities = []

    for obj in objects:
        meshes.append(build_mesh(obj))

        velocities.append(
            load_tensor(
                obj.get("velocity", [0.0, 0.0, 0.0]),
                device=device,
            )
        )

    voxels, node_groups = Voxels.from_meshes(
        meshes,
        h=scene["voxels"]["h"],
    )

    voxels.init_state(
        density=scene["voxels"].get("density", 1.0)
    )

    for nodes, vel in zip(node_groups, velocities):
        voxels.node_vel[nodes] = vel

    sim = Simulation(
        voxels        = voxels,
        k             = scene["simulation"]["k"],
        dt            = 1 / scene["simulation"]["dt-1"],
        ground_y      = scene["simulation"].get("ground_y", 0.0),
        self_collide  = scene["simulation"].get("self_collide", True),
        do_fracture   = scene["simulation"].get("do_fracture", False),
        tensile_yield = scene["simulation"].get("tensile_yield", 0.1),
    )

    return sim, scene.get("title", "simulation")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python main.py <scene.json>")
        sys.exit(1)

    scene_path = sys.argv[1]

    sim, title = load_scene(scene_path)

    print(
        f"V={sim.voxels.V}  "
        f"N={sim.voxels.N}  "
        f"E={sim.voxels.E}"
    )

    window.run(sim, title=title)