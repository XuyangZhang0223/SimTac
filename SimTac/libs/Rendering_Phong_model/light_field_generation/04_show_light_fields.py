import os
from math import pi, sin, cos

import numpy as np

from trimesh.exchange.load import load

from sim_model.model import normalize_vectors, partial_derivative
from sim_model.model_save import show_field
import cv2

from sim_model.utils.camera import get_camera_matrix, depth2cloud

# assets_path = os.path.dirname(os.path.abspath(__file__)) + '/../assets/'

__location__ = os.path.dirname(os.path.abspath(__file__))

from sim_model.utils.maths import norm_vectors

assets_path = os.path.join(__location__, 'experimental_setup/geltip/sim_assets')

# cloud_size = (32, 24)
# cloud_size = (64, 48)
cloud_size = (160, 120)
#cloud_size = (640, 480)

#method = 'geodesic'
#method = 'rtransport'
method = 'linear'
#method = 'plane'

prefix = str(cloud_size[0]) + 'x' + str(cloud_size[1])
# cloud = np.load(assets_path + '/' + prefix + '_ref_cloud.npy')

#mesh = load(assets_path + '/' + prefix + '_aligned_mesh.stl', 'stl', force='mesh')
mesh_file_path = os.path.join(assets_path, prefix + '_aligned_mesh.stl')
with open(mesh_file_path, 'rb') as f:
    mesh = load(f, file_type='stl', force='mesh')
scale = 0.001
# z = -0.015
led_height = 0
led_radius = 15.5
# leds = [
#     np.array([
#         cos(a * (2 * pi / 3)) * scale * led_radius,
#         sin(a * (2 * pi / 3)) * scale * led_radius,
#         z
#     ])
#     for a in range(3)
# ]
a = 6*pi/9
leds = [
    np.array([cos((pi / 3) + a) * led_radius, sin((pi / 3) + a) * led_radius, led_height]),  # Red LED
    np.array([cos((pi) + a) * led_radius, sin((pi) + a) * led_radius, led_height]),  # Green LED
    np.array([cos((5 * pi / 3) + a) * led_radius, sin((5 * pi / 3) + a) * led_radius, led_height])  # Blue LED
  ]
leds_colors = ['red', 'green', 'blue']
print(leds)

depth = np.load(assets_path + '/bkg_for_light_field_generation.npy')
depth = cv2.resize(depth, cloud_size, interpolation=cv2.INTER_LINEAR)
cam_matrix = get_camera_matrix(depth.shape[::-1], fov_deg=100)
cloud = depth2cloud(cam_matrix, depth)
cloud_map = cloud.reshape((cloud_size[1], cloud_size[0], 3))

dx = normalize_vectors(partial_derivative(cloud_map, 'x'))
dy = normalize_vectors(partial_derivative(cloud_map, 'y'))
normals = np.cross(dx, dy)



for l in range(len(leds)):
    
    field = np.load(assets_path + '/' + method + '_' + prefix + '_field_' + str(l) + '.npy', allow_pickle=True)
    print(field.shape)
    print(field.dtype)
    # print(np.mean(norm_vectors(normalize_vectors(field))))

    #show_field(cloud_map=cloud_map, field=field, field_color=leds_colors[l], mesh=mesh, pts=[leds[l]], subsample=25)
    show_field(cloud_map=cloud_map, field=field, field_color=leds_colors[l], mesh=mesh, subsample=62)

