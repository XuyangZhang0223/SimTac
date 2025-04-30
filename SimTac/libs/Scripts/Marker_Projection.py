import numpy as np
import cv2
from PIL import Image, ImageDraw

camera_position = np.array([20, 20, 5.28])  
fov = 99
image_width = 640
image_height = 480

# image_width = 1280
# image_height = 960

def project_points(point_cloud, camera_position, image_width, image_height, fov, rotation_angle):
    projected_points = []
    focal_length = (image_width / 2) / np.tan(np.deg2rad(fov / 2))
    rotation_matrix = np.array([[np.cos(rotation_angle), np.sin(rotation_angle)],
                                [-np.sin(rotation_angle), np.cos(rotation_angle)]])
    for point in point_cloud:
        vec_to_point = point - camera_position
        distance = np.linalg.norm(vec_to_point)
        if distance > 0:
            projected_point = vec_to_point[:2]
            projected_point = np.dot(rotation_matrix, projected_point)
            projected_x = int((projected_point[1] / vec_to_point[2]) * focal_length + (image_width / 2))
            projected_y = int((projected_point[0] / vec_to_point[2]) * focal_length + (image_height / 2))
            projected_points.append((projected_x, projected_y))
            
    return projected_points

def combine_image(initial_npy, image_b, displacement, rotation_angle):
    real_npy = initial_npy + displacement
    rotation_angle = np.deg2rad(rotation_angle) 
    projected_points = project_points(real_npy, camera_position, image_width, image_height, fov, rotation_angle)
    image_b = np.array(image_b)
    image_b = np.squeeze(image_b)
    for projected_point, initial_point, disp in zip(projected_points, real_npy, displacement):
        distance_to_camera = np.linalg.norm(initial_point - camera_position)
        distance_z = initial_point[2] - camera_position[2]
        if 0 <= distance_z <= (32.8 - camera_position[2]):
            scale_factor = 7*(42-distance_z)/42  
            #scale_factor = 11*(42-distance_z)/42  # 1280 x 960
            radius = max(0.5, int(scale_factor))
            cv2.circle(image_b, projected_point, radius, (40, 40, 40), -1)

    return image_b
