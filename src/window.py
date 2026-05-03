import time
import torch
import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from voxels import Voxels
from simulation import Simulation

class Camera:
    def __init__(self, distance=20.0, yaw=45.0, pitch=30.0, target=(0, 0, 0)):
        self.distance = distance
        self.yaw = yaw
        self.pitch = -pitch
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

def _init_gl(width, height, far=2000.0):
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
    gluPerspective(45, width / height, 0.1, far)
    glMatrixMode(GL_MODELVIEW)

def _draw_floor(height: float, size=100, color=(1.0, 0.7, 0.5)):
    y = height

    verts = torch.tensor([
        [-size, y, -size],
        [ size, y, -size],
        [ size, y,  size],
        [-size, y,  size],
    ], dtype=torch.float32)

    normals = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=torch.float32)

    verts = verts.reshape(-1).cpu().numpy()
    normals = normals.reshape(-1).cpu().numpy()

    glColor3f(*color)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, verts)
    glNormalPointer(GL_FLOAT, 0, normals)
    glDrawArrays(GL_QUADS, 0, 4)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)

def _draw_voxels(voxels: Voxels, color=(0.6, 0.7, 0.8)):
    face_nodes, normals = voxels.boundary_faces()
    if face_nodes.shape[0] == 0:
        return
    verts = voxels.node_pos[face_nodes].reshape(-1, 3).cpu().numpy()
    normals = normals.repeat_interleave(4, dim=0).cpu().numpy()

    glColor3f(*color)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, verts)
    glNormalPointer(GL_FLOAT, 0, normals)
    glDrawArrays(GL_QUADS, 0, len(verts))
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)


def _draw_edges(voxels: Voxels, color=(0.0, 0.0, 0.0)):
    face_nodes, _ = voxels.boundary_faces()
    if face_nodes.shape[0] == 0:
        return
    verts = voxels.node_pos[face_nodes].cpu()
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


def run(sim: Simulation, title: str = "voxels", size=(800, 800)):
    pg.init()
    pg.display.set_mode(size, DOUBLEBUF | OPENGL)
    pg.display.set_caption(title)

    voxels = sim.voxels
    extent = (voxels.node_pos.max() - voxels.node_pos.min()).item()
    _init_gl(*size, far=max(2000.0, extent * 8.0))

    center = voxels.node_pos.mean(dim=0).tolist()
    cam = Camera(distance=extent * 2.0, target=center)

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
            t0 = time.time()
            n_broken = sim.step()
            step_ms = (time.time() - t0) * 1000
            if n_broken > 0:
                print(f"step {step_ms:.0f}ms  broke {n_broken} links  V={sim.voxels.V}  N={sim.voxels.N}")

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        cam.apply()

        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        _draw_voxels(sim.voxels)
        _draw_floor(sim.ground_y)
        glDisable(GL_POLYGON_OFFSET_FILL)
        _draw_edges(sim.voxels)

        pg.display.flip()
        clock.tick(60)
