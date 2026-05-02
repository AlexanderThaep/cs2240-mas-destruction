import torch
import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from voxels import Voxels
from physics import Physics
from fracture import break_yielded_links
from collision import ground_collision_forces, voxel_voxel_collision_forces


class Camera:
    def __init__(self, distance=20.0, yaw=45.0, pitch=30.0, target=(0, 0, 0)):
        self.distance = distance
        self.yaw = yaw
        self.pitch = pitch
        self.target = list(target)

    def apply(self):
        glLoadIdentity()
        gluLookAt(0, 0, self.distance, 0, 0, 0, 0, 1, 0)
        glRotatef(-self.pitch, 1, 0, 0)
        glRotatef(-self.yaw,   0, 1, 0)
        glTranslatef(-self.target[0], -self.target[1], -self.target[2])

    def orbit(self, dx, dy):
        self.yaw   += dx * 0.3
        self.pitch += dy * 0.3
        self.pitch = max(-89, min(89, self.pitch))

    def zoom(self, delta):
        self.distance *= 0.9 if delta > 0 else 1.1
        self.distance = max(1.0, self.distance)


def _init_gl(width, height):
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glClearColor(0.15, 0.15, 0.18, 1.0)

    glLight(GL_LIGHT0, GL_POSITION, (0.3, 1.0, 0.5, 0.0))
    glLight(GL_LIGHT0, GL_DIFFUSE,  (1.0, 1.0, 1.0, 1.0))
    glLight(GL_LIGHT0, GL_AMBIENT,  (0.2, 0.2, 0.2, 1.0))

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / height, 0.1, 500.0)
    glMatrixMode(GL_MODELVIEW)


def _draw_scene(scene: Voxels, color=(0.6, 0.7, 0.8)):
    face_nodes, normals = scene.boundary_faces()
    if face_nodes.shape[0] == 0:
        return
    verts = scene.node_pos[face_nodes].reshape(-1, 3).cpu().numpy()
    normals = normals.repeat_interleave(4, dim=0).cpu().numpy()

    glColor3f(*color)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, verts)
    glNormalPointer(GL_FLOAT, 0, normals)
    glDrawArrays(GL_QUADS, 0, len(verts))
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)


def _draw_edges(scene: Voxels, color=(0.0, 0.0, 0.0)):
    face_nodes, _ = scene.boundary_faces()
    if face_nodes.shape[0] == 0:
        return
    verts = scene.node_pos[face_nodes].cpu()     # (F,4,3)
    v0, v1, v2, v3 = verts[:, 0], verts[:, 1], verts[:, 2], verts[:, 3]
    edges = torch.stack([v0, v1, v1, v2, v2, v3, v3, v0], dim=1).reshape(-1, 3).numpy()

    glDisable(GL_LIGHTING)
    glColor3f(*color)
    glLineWidth(1.0)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, edges)
    glDrawArrays(GL_LINES, 0, len(edges))
    glDisableClientState(GL_VERTEX_ARRAY)
    glEnable(GL_LIGHTING)


def run(
    physics: Physics,
    dt: float = 1/60,
    title: str = "voxels",
    size=(800, 800),
    tensile_yield: float = 0.15,
    ground_y: float = -5.0,
    enable_fracture: bool = True,
    enable_self_collision: bool = True,
):
    pg.init()
    pg.display.set_mode(size, DOUBLEBUF | OPENGL)
    pg.display.set_caption(title)
    _init_gl(*size)

    scene = physics.scene
    center = scene.node_pos.mean(dim=0).tolist()
    span = (scene.node_pos.max() - scene.node_pos.min()).item()
    cam = Camera(distance=span * 2.0, target=center)

    clock = pg.time.Clock()
    dragging = False
    paused = False

    while True:
        for ev in pg.event.get():
            if ev.type == QUIT or (ev.type == KEYDOWN and ev.key == K_ESCAPE):
                pg.quit()
                return
            elif ev.type == KEYDOWN and ev.key == K_SPACE:
                paused = not paused
            elif ev.type == MOUSEBUTTONDOWN and ev.button == 1:
                dragging = True
            elif ev.type == MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == MOUSEMOTION and dragging:
                cam.orbit(ev.rel[0], ev.rel[1])
            elif ev.type == MOUSEWHEEL:
                cam.zoom(ev.y)

        if not paused:
            f_ext = ground_collision_forces(scene, ground_y=ground_y)
            if enable_self_collision:
                components = scene.connected_components()
                f_ext = f_ext + voxel_voxel_collision_forces(scene, components)
            physics.step(dt, external_forces=f_ext)

            if enable_fracture:
                n_broken = break_yielded_links(scene, tensile_yield=tensile_yield)
                if n_broken > 0:
                    print(f"fractured {n_broken} link(s) -> {scene.connected_components().max().item() + 1} components")

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        cam.apply()

        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        _draw_scene(scene)
        glDisable(GL_POLYGON_OFFSET_FILL)
        _draw_edges(scene)

        pg.display.flip()
        clock.tick(60)
