import math
import torch
import numpy as np
import open3d as o3d
import cv2

# def circle_mask(size=(64, 48), border=0, channels=0, value=1):
#     """
#     Generate a circular mask for a given image size,
#     with both inside and outside pixels having the same value.
#     """
#     if channels == 0:
#         m = torch.ones((size[1], size[0]), dtype=torch.float32) * value  # Initialize mask with the specified value
#         m_center = (size[0] // 2, size[1] // 2)
#         m_radius = min(size[0], size[1]) // 2 - border
        
#         # Use OpenCV to draw the circle
#         m_np = m.cpu().numpy()
#         cv2.circle(m_np, m_center, m_radius, value, -1)
#         m = torch.tensor(m_np, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
#         return m
    
#     return torch.stack([circle_mask(size, border, channels=0, value=value) for _ in range(channels)], dim=2)

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

# we use cpu here since we need open3d fuction which can not be transfered to torch
def depth2cloud(cam_matrix, depth):
    if isinstance(depth, torch.Tensor):
        # Move tensor to CPU if it's on GPU
        if depth.is_cuda:
            depth = depth.cpu()

        # Convert tensor to numpy array
        depth = depth.numpy()
    
    print(depth.shape)
    invalid_depth = np.isnan(depth) | (depth < 0)
    print(f"Number of invalid depth values: {np.sum(invalid_depth)}")

    # Get the point cloud from the depth image
    point_cloud = get_cloud_from_depth(cam_matrix, depth)
    
    # Access the points from the Open3D PointCloud object
    points = np.asarray(point_cloud.points)
    #print(len(points))
    
    # Convert points to a PyTorch tensor if needed
    points_tensor = torch.tensor(points, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

    return points_tensor.reshape((depth.shape[0], depth.shape[1], 3))
