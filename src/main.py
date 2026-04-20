from voxels import VoxelMesh
import window

import torus_15x5x15

# This is just a toy example to see if fracturing is working like we'd expect.
# Rn it takes a 6x6x6 cube and chops it into two 3x6x6 halves.

mesh = VoxelMesh.cube(side=6, h=1.0)
mesh = VoxelMesh.from_py(torus_15x5x15, h=1.0)
print(f"voxels: {mesh.V}   nodes: {mesh.N}")
print(f"components: {mesh.connected_components().max().item() + 1}")

# # Break slice to create fracture
# for v in range(mesh.V):
#     gx = mesh.voxel_coords[v, 0].item()
#     for d in range(6):
#         j = mesh.links[v, d].item()
#         if j < 0:
#             continue
#         gx_j = mesh.voxel_coords[j, 0].item()
#         # Sever all links crossing the x=3 plane
#         if (gx < 3 and gx_j >= 3) or (gx >= 3 and gx_j < 3):
#             mesh.break_link(v, j)

# mesh.rebuild_after_fracture()
# print(f"after fracture — nodes: {mesh.N}")
# print(f"components: {mesh.connected_components().max().item() + 1}")

# # Nudge right half so split is visible
# right_nodes = set()
# for v in range(mesh.V):
#     if mesh.voxel_coords[v, 0].item() >= 3:
#         right_nodes.update(mesh.voxel_nodes[v].tolist())
# for nid in right_nodes:
#     mesh.node_pos[nid, 0] += 1.5

# window.run(mesh)
window.run_scene([mesh])