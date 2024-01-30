import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
vid = cv2.VideoCapture(0)
# Window dimensions
window_width, window_height = 800, 600

# Rotation angles
x_rot = y_rot = z_rot = 0.0

def init_glut():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(window_width, window_height)
    glutInitWindowPosition(200, 200)
    glutCreateWindow("OpenGL Plane with Camera Rotation")
    glutDisplayFunc(draw_scene)
    glutIdleFunc(draw_scene)
    glutKeyboardFunc(keyboard)

def init_opengl():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)

def draw_scene():
    global x_rot, y_rot, z_rot
    ret, frame = vid.read()
    frame = frame/255
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluPerspective(45, (window_width / window_height), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    glRotatef(x_rot, 1.0, 0.0, 0.0)
    glRotatef(y_rot, 0.0, 1.0, 0.0)
    glRotatef(z_rot, 0.0, 0.0, 1.0)

    # Draw a simple plane
    glBegin(GL_POINTS)
    for i in range(224):
        for j in range(224):
            glColor3f(frame[i][j][2], frame[i][j][1], frame[i][j][0])
            glVertex3f(i*0.01 - 1, j*0.01 - 1,  frame[i][j][2])
    
    glEnd()

    glutSwapBuffers()

def keyboard(key, x, y):
    global x_rot, y_rot, z_rot

    if key == b'w':
        x_rot -= 5.0
    elif key == b's':
        x_rot += 5.0
    elif key == b'a':
        y_rot -= 5.0
    elif key == b'd':
        y_rot += 5.0

def main():
    init_glut()
    init_opengl()
    glutMainLoop()

if __name__ == "__main__":
    main()
