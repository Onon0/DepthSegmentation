import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2

import torch
import torch.nn as nn
from torchvision import datasets, models
import torch.nn.functional as F

vid = cv2.VideoCapture(0)
# Window dimensions
window_width, window_height = 800, 600

# Rotation angles
x_rot = y_rot = z_rot = 0.0



class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.convA( torch.cat([up_x, concat_with], dim=1)  ) )  )

class Decoder(nn.Module):
    def __init__(self, num_features=1664, decoder_width = 1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features//1 + 256, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 128,  output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 64,  output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 + 64,  output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[12]
        x_d0 = self.conv2(F.relu(x_block4))

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()       
        self.original_model = models.densenet169( weights=True )

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items(): features.append( v(features[-1]) )
        return features

class PTModel(nn.Module):
    def __init__(self):
        super(PTModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder( self.encoder(x) )
      #  return self.encoder(x)

model = PTModel()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model.load_state_dict(torch.load(".\\2023-12-26_07-38_0.02814927462066213_30.pth"))
print(DEVICE)
model.to(DEVICE)
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
    frame = frame[0:224,0:224]
    frame = frame/255

    depth = torch.tensor(frame).permute(2,1,0).unsqueeze(0).to(device = DEVICE)

    depth = model(depth.float())
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
            
            glVertex3f(i*0.01 - 1, j*0.01 - 1,  depth[0][0][int(i/2)][int(j/2)]*2)
    
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
