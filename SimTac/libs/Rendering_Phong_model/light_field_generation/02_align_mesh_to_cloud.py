## notes: Align the mesh (stl file) of the membrane with the bkg.npy depth map through scale, trasnlation and rotation
## if we have the membrane mesh and corresponding bkg.npy file, there is no need to align them.

import os
from math import pi

import cv2
import numpy as np

from trimesh.exchange.load import load
from trimesh.transformations import scale_matrix, translation_matrix, rotation_matrix
from trimesh import Scene

from sim_model.utils.vis_mesh import sphere
import open3d as o3d

__location__ = os.path.dirname(os.path.abspath(__file__))

from sim_model.utils.camera import get_camera_matrix, depth2cloud, circle_mask, get_cloud_from_depth

assets_path = os.path.join(__location__, 'experimental_setup/geltip/sim_assets')

# mesh_path = assets_path + '/../../meshes/elastomer_very_long.stl'
mesh_path = assets_path + '/elastomer.stl'

# cloud_size = (32, 24)
# cloud_size = (64, 48)
# cloud_size = (320, 240)
#cloud_size = (160, 120)
cloud_size = (640, 480)
fov = 100

depth = np.load(assets_path + '/bkg.npy')
depth = cv2.resize(depth, cloud_size, interpolation=cv2.INTER_LINEAR)
cam_matrix = get_camera_matrix(depth.shape[::-1], fov_deg=fov)

prefix = str(cloud_size[0]) + 'x' + str(cloud_size[1])
cloud = get_cloud_from_depth(cam_matrix, depth)
o3d.visualization.draw_geometries([cloud])
cloud = cloud.points

#mesh = load(mesh_path, 'stl', force='mesh')
mesh_file = open(mesh_path, 'rb')  # 打开STL文件
mesh = load(mesh_file, 'stl', force='mesh')  # 将文件对象传递给load函数
mesh_file.close()  # 关闭文件

# TODO: mesh is visually aligned.
# this should be improved in the future.
# trans, cost = mesh.register(cloud); didn't work
x_trans = 0.015
y_trans = -0.015
z_trans = -0.0043
translation_vector = np.array([x_trans, y_trans, z_trans])
translation_matrix = translation_matrix(translation_vector)
mesh = mesh.apply_transform(scale_matrix(0.001))
mesh = mesh.apply_transform(rotation_matrix(pi / 2, np.array([0, 0, 1])))
mesh = mesh.apply_transform(translation_matrix)

mesh.export(f'{assets_path}/{prefix}_aligned_mesh.stl')
print('aligned.')

mesh.visual.face_colors = [200, 200, 200, 200]
spheres = [sphere(cloud[i]) for i in range(len(cloud)) if i % 100 == 0]
scene = Scene([mesh, *spheres])
scene.show()
