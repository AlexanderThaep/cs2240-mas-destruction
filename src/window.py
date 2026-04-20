import torch
from quatorch import Quaternion
import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from voxels import VoxelMesh
import rigid

class Camera:
    def __init__(self, distance=20.0, yaw=45.0, pitch=30.0, target=(0,0,0)):
        self.distance = distance
        self.yaw   = yaw
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


def init_gl(width, height):
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


def draw_mesh(mesh: VoxelMesh, color=(0.6, 0.7, 0.8)):
    """Draw all boundary faces as flat-shaded quads."""
    verts, normals = mesh.boundary_faces()   # (F,4,3), (F,3)

    verts = verts.reshape(-1, 3).cpu().numpy()
    normals = normals.repeat_interleave(4, dim=0).cpu().numpy()

    glColor3f(*color)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)

    glVertexPointer(3, GL_FLOAT, 0, verts)
    glNormalPointer(GL_FLOAT, 0, normals)

    glDrawArrays(GL_QUADS, 0, len(verts))

    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)


def draw_edges(mesh: VoxelMesh, color=(0.0, 0.0, 0.0)):
    """Draw boundary face edges as wireframe overlay."""
    verts, _ = mesh.boundary_faces()   # (F,4,3)
    verts = verts.cpu()

    # build edge list in tensor form
    v0 = verts[:, 0]
    v1 = verts[:, 1]
    v2 = verts[:, 2]
    v3 = verts[:, 3]

    edges = torch.stack([
        v0, v1,
        v1, v2,
        v2, v3,
        v3, v0
    ], dim=1).reshape(-1, 3).numpy()   # (F*8, 3)

    glDisable(GL_LIGHTING)
    glColor3f(*color)
    glLineWidth(1.0)

    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, edges)

    glDrawArrays(GL_LINES, 0, len(edges))

    glDisableClientState(GL_VERTEX_ARRAY)
    glEnable(GL_LIGHTING)

def run(mesh: VoxelMesh, title="voxels", size=(800, 800)):
    pg.init()
    pg.display.set_mode(size, DOUBLEBUF | OPENGL)
    pg.display.set_caption(title)
    init_gl(*size)

    # center camera on mesh
    center = mesh.node_rest.mean(dim=0).tolist()
    span   = (mesh.node_rest.max() - mesh.node_rest.min()).item()
    cam    = Camera(distance=span * 1.8, target=center)

    clock = pg.time.Clock()
    dragging = False

    com = rigid.compute_voxel_com(mesh.node_rest)
    M = 0.2
    I = rigid.compute_voxel_inertia(com, mesh.voxel_coords, M)
    V = torch.tensor([0.0, 0.0, 0.0])
    W = torch.tensor([0.0, 0.0, 0.0])

    new_com = com.clone()

    tq = torch.tensor([0.0, 0.8, 0.4])
    f = torch.tensor([0.0, 0.0, 0.0])

    while True:
        for ev in pg.event.get():
            if ev.type == QUIT or (ev.type == KEYDOWN and ev.key == K_ESCAPE):
                pg.quit()
                return
            elif ev.type == MOUSEBUTTONDOWN and ev.button == 1:
                dragging = True
            elif ev.type == MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == MOUSEMOTION and dragging:
                cam.orbit(ev.rel[0], ev.rel[1])
            elif ev.type == MOUSEWHEEL:
                cam.zoom(ev.y)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        cam.apply()

        new_com, V, orientation, W = rigid.integrate_rigid_body(com, V, W, rigid.Q_IDENTITY, I, tq, f, M, 1 / 60)
        mesh.node_rest = rigid.apply_voxels(mesh.node_rest, com, new_com, orientation)
        com = new_com

        # Add a slight polygon offset so edges sit on top of faces
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        draw_mesh(mesh)
        glDisable(GL_POLYGON_OFFSET_FILL)
        draw_edges(mesh)

        pg.display.flip()
        clock.tick(60)