import pygame as pg
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

def main():
    pg.init()
    windowSize = (800, 800)
    pg.display.set_mode(windowSize, DOUBLEBUF|OPENGL)

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        pg.display.flip()
        pg.time.wait(10)

if __name__ == "__main__":
    main()