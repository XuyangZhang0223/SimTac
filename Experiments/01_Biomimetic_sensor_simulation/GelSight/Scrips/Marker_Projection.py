import numpy as np
import cv2
from PIL import Image, ImageDraw

camera_position = np.array([20, 20, -5])  
fov = 72
image_width = 640
image_height = 480

def project_points(point_cloud, camera_position, image_width, image_height, fov):
    projected_points = []
    focal_length = (image_width / 2) / np.tan(np.deg2rad(fov / 2))

    for point in point_cloud:
        vec_to_point = point - camera_position
        distance = np.linalg.norm(vec_to_point)

        if distance > 0:
            projected_point = vec_to_point[:2]

            projected_x = int((projected_point[0] / vec_to_point[2]) * focal_length + (image_width / 2))
            projected_y = int((projected_point[1] / vec_to_point[2]) * focal_length + (image_height / 2))

            projected_points.append((projected_x, projected_y))

    return projected_points

# pay attention the initial_npy here has already + np.array([20, 20, 5]).
def combine_image(initial_npy, image_b, displacement):

    real_npy = initial_npy + displacement

    print(real_npy)

    projected_points = project_points(real_npy, camera_position, image_width, image_height, fov)

    print(projected_points)

    image_b = np.array(image_b)
    image_b = np.squeeze(image_b)

    for projected_point in projected_points:

        radius = 8

        cv2.circle(image_b, projected_point, radius, (40, 40, 40), -1)


    return image_b
