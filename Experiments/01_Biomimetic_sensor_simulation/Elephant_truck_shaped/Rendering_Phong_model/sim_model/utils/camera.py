import math
import open3d as o3d

import cv2
import numpy as np


def circle_mask(size=(64, 48), border=0, channels=0, value=1):
    """
    Used to generate a circular mask for a given image size,
    with both inside and outside pixels having the same value.
    """
    if channels == 0:
        m = np.ones((size[1], size[0])) * value  # Initialize mask with the specified value
        m_center = (size[0] // 2, size[1] // 2)
        m_radius = min(size[0], size[1]) // 2 - border
        cv2.circle(m, m_center, m_radius, value, -1)  # Fill the circle region with the specified value
        return m.astype(np.float32)
    return np.stack([circle_mask(size, border, channels=0, value=value) for _ in range(channels)], axis=2)

def get_camera_matrix(img_size, fov_deg):
    img_width, img_height = img_size

    fov = math.radians(fov_deg)
    f = img_height / (2 * math.tan(fov / 2))
    cx = (img_width - 1) / 2
    cy = (img_height - 1) / 2

    return o3d.camera.PinholeCameraIntrinsic(img_width, img_height, f, f, cx, cy)


def get_cloud_from_depth(cam_matrix, depth):
    o3d_depth = o3d.geometry.Image(depth)
    o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, cam_matrix)
    return o3d_cloud


def depth2cloud(cam_matrix, depth):
    print(depth.shape)
    invalid_depth = np.isnan(depth) | (depth < 0)
    print(f"Number of invalid depth values: {np.sum(invalid_depth)}")
    points = get_cloud_from_depth(cam_matrix, depth).points
    print(len(points))
    return np.array(points).reshape((depth.shape[0], depth.shape[1], 3))
