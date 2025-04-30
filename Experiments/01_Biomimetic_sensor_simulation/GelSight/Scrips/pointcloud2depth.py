import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from depth_generation import generate
from GelTip_Rendering import generate_tactile_rgb
import cv2
import os
import time
import cv2
tic = time.time()

Contact_Surface_set_path = "./Abaqus_model/MESH04/Contact_Surface_index.txt"
Contact_Surface_list = []
with open(Contact_Surface_set_path, "r") as file:
    for line in file:
        data_1 = int(line.strip())  
        Contact_Surface_list.append(int(data_1))
Contact_Surface_set_num = len(Contact_Surface_list)

#define the camera
fov_degrees = 120
width = 40
height = 40
camera_position = np.array([0, 0, 5])  
fx = width / (2 * np.tan(np.radians(fov_degrees / 2)))  
fy = fx 
cx = width / 2  
cy = height / 2

img_length = 640
img_width = 480

i_depth = np.zeros((img_width,img_length))
u = np.zeros(Contact_Surface_set_num)
v = np.zeros(Contact_Surface_set_num)
depth = np.zeros(Contact_Surface_set_num)

xi = np.linspace(0, 40, num=img_length)
yi = np.linspace(5,35, num=img_width)
width = 640  
height = 480
depth_map = np.zeros((height, width), dtype=np.float32)
k=1

data = np.load('initial_npy.npy')

def wide_angle_camera_transfer(xp, yp, p_depth):
  for i in range(len(xp)):

    relative_position = np.array([xp[i], yp[i], p_depth[i]]) - camera_position

    depth[i] = np.linalg.norm(relative_position)

    u[i] = fx * relative_position[0] / depth[i] + cx
    v[i] = fy * relative_position[1] / depth[i] + cy
    
    #depth_map[int(v[i]), int(u[i])] = depth[i]/1000

  return u, v, depth, #depth_map
  
xp = data[:, 0]
yp = data[:, 1]
p_depth = data[:, 2]
print(xp.shape)
    
wide_angle_camera_transfer(xp, yp, p_depth)

    
i_depth = interpolate.griddata((v, u), depth/1000, (xi[None,:],yi[:,None]), method='cubic', fill_value=0.001)

i_depth = i_depth.astype(np.float32)

npy_name = "Results" + "/MESH04/" + "haha.npy"

np.save(npy_name,i_depth)


    
    
    
