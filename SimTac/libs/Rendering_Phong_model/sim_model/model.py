#!/usr/bin/env python
import cv2
import numpy as np
import time
import open3d as o3d

from .utils.camera import get_camera_matrix, depth2cloud
from .utils.maths import normalize_vectors, gkern2, dot_vectors, normals, proj_vectors, partial_derivative
from .utils.vis_img import to_normed_rgb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def adjust_luminance(r, g, b, shadow_factor):
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    adjusted_r = r * (1 - shadow_factor) + luminance * shadow_factor
    adjusted_g = g * (1 - shadow_factor) + luminance * shadow_factor
    adjusted_b = b * (1 - shadow_factor) + luminance * shadow_factor
    return adjusted_r, adjusted_g, adjusted_b

def apply_soft_shadow(image, shadow_factor, kernel_size=43):
    image = image.astype(np.float32)
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    shadow_image = image * (1 - shadow_factor) + blurred_image * shadow_factor
    shadow_image = np.clip(shadow_image, 0, 255)
    shadow_image = shadow_image.astype(np.uint8)
    
    return shadow_image

def visualize_point_cloud(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates from points array
    x = points[:, :, 0].flatten()
    y = points[:, :, 1].flatten()
    z = points[:, :, 2].flatten()
    
    # Plot point cloud
    ax.scatter(x, y, z, marker='.', color='b', s=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

class SimulationModel:

    def __init__(self, **config):
        self.default_ks = 0.15
        self.default_kd = 0.5
        self.default_alpha = 100
        self.ia = config['ia'] or 1
        self.fov = config['fov'] or 100

        self.lights = config['light_sources']
        self.rectify_fields = config['rectify_fields']

        self.bkg_depth = config['background_depth']
        self.cam_matrix = get_camera_matrix(self.bkg_depth.shape[::-1], self.fov)

        self.background_img = config['background_img']
        self.s_ref = depth2cloud(self.cam_matrix, self.bkg_depth)  # config['cloud_map']
        self.s_ref_n = normals(self.s_ref)

        self.apply_elastic_deformation = config['elastic_deformation'] if 'elastic_deformation' in config else False
        self.internal_shadow = config['internal_shadow'] if 'internal_shadow' in config else 0.15

        # pre compute & defaults
        self.ambient = config['background_img']

        for light in self.lights:
            light['ks'] = light['ks'] if 'ks' in light else self.default_ks
            light['kd'] = light['kd'] if 'kd' in light else self.default_kd
            light['alpha'] = light['alpha'] if 'alpha' in light else self.default_alpha

            light['color_map'] = np.tile(np.array(np.array(light['color']) / 255.0)
                                         .reshape((1, 1, 3)), self.s_ref.shape[0:2] + (1,))

        self.texture_sigma = config['texture_sigma'] or 0.00001
        self.t = config['t'] if 't' in config else 3
        self.sigma = config['sigma'] if 'sigma' in config else 7
        self.kernel_size = config['sigma'] if 'sigma' in config else 21

    @staticmethod
    def load_assets(assets_path, input_res, output_res, lf_method, n_light_sources):
        prefix = str(input_res[1]) + 'x' + str(input_res[0])

        light_fields = [
            cv2.resize(
                cv2.GaussianBlur(
                    np.load(assets_path + '/' + lf_method + '_' + prefix + '_field_' + str(l) + '.npy'),
                    (25, 25), 0),
                output_res, interpolation=cv2.INTER_LINEAR)
            # )
            for l in range(n_light_sources)
        ]
        # normals,
        return light_fields

    def gauss_texture(self, shape):
        row, col = shape
        mean = 0
        gauss = np.random.normal(mean, self.texture_sigma, (row, col))
        gauss = gauss.reshape(row, col)
        return np.stack([gauss, gauss, gauss], axis=2)

    def _spec_diff(self, lm_data, v, n, s):
        imd = lm_data['id']
        ims = lm_data['is']
        alpha = lm_data['alpha']

        lm = - lm_data['field']
        color = lm_data['color_map']

        if self.rectify_fields:
            lm = normalize_vectors(lm - proj_vectors(lm, self.s_ref_n))

        # Shared calculations
        lm_n = dot_vectors(lm, n)
        lm_n[lm_n < 0.0] = 0.0
        Rm = 2.0 * lm_n[:, :, np.newaxis] * n - lm

        # diffuse component
        diffuse_l = lm_n * imd

        # specular component
        spec_l = (dot_vectors(Rm, v) ** alpha) * ims

        return (diffuse_l + spec_l)[:, :, np.newaxis] * color

    def calculate_occluded_areas(self, protrusion_map, optical_rays):
        # Threshold the protrusion_map to create a binary map, True = 1, False = 0
        binary_map = (protrusion_map > 0.00001).astype(np.float32)

        binary_map = cv2.GaussianBlur(binary_map, (5, 5), 1)

        kernel = np.ones((13, 13), np.uint8)
        binary_map = cv2.erode(binary_map, kernel, iterations=1)

        binary_map = cv2.GaussianBlur(binary_map, (5, 5), 1)

        # that should be adapted to the contact depth
        # cv2.imshow('Smoothed Binary Map', binary_map * 255)  # 乘以255以便于显示
        # cv2.waitKey(0)
        # cv2.imshow('Smoothed Binary Map', binary_map * 255)  # 乘以255以便于显示
        # cv2.waitKey(0)

        # 将模糊图像阈值化为二值图像
        #smoothed_binary_map = (blurred_map > 0.5).astype(np.uint8)  # 0.5 是阈值，可以根据需要调整

        # Compute the partial derivatives of the binary map
        areas_x = partial_derivative(binary_map, 'x')
        areas_y = partial_derivative(binary_map, 'y')

        # Compare the signs of the optical rays and partial derivatives
        sign_comparison = np.equal(np.sign(optical_rays[:, :, :2]), np.sign(np.stack([areas_x, areas_y], axis=-1)))

        # Calculate the occluded areas
        occluded_areas = np.clip(sign_comparison.sum(axis=-1) / 0.05, 0, 1)

        # Dilate the occluded areas, that will affect the size of the shadow area.
        kernel = np.ones((3, 3), np.uint8)
        occluded_areas = cv2.dilate(occluded_areas, kernel, iterations=1)

        # Apply a Gaussian filter
        occluded_areas = cv2.filter2D(occluded_areas, -1, gkern2(55, 5))

        # Normalize the occluded areas
        occluded_areas = (occluded_areas - occluded_areas.min()) / (occluded_areas.max() - occluded_areas.min())

        # Remove regions where the binary map has a value of 1
        # kernel = np.ones((1, 1), np.uint8)
        # dilated_binary_map = cv2.erode(binary_map, kernel, iterations=1)
        occluded_areas *= (1 - binary_map)

        return occluded_areas

    def calculate_occluded_areas_alternative(self, surface_normals, optical_rays, threshold=0.95):
        # Compute the dot product between surface normals and optical rays
        dot_product = np.abs(np.sum(surface_normals * optical_rays, axis=-1))

        # Threshold the dot product to create an occlusion map
        occlusion_map = (dot_product > threshold).astype(np.float32)

        # Dilate the occlusion map
        kernel = np.ones((3, 3), np.uint8)
        occlusion_map = cv2.dilate(occlusion_map, kernel, iterations=1)

        # Apply a Gaussian filter
        occlusion_map = cv2.GaussianBlur(occlusion_map, (55, 55), 5)

        # Normalize the occlusion map
        occlusion_map = (occlusion_map - occlusion_map.min()) / (occlusion_map.max() - occlusion_map.min())

        return occlusion_map
    

    def generate(self, depth):
        # 记录函数开始时间
        start_time = time.time()

        # 计算 protrusion_map, 计算深度差的图像
        start = time.time()
        protrusion_map = self.bkg_depth - depth
        end = time.time()
        print(f"Protrusion map computation time: {end - start:.4f} seconds")

        # 计算二值图像，这是接触区域的外边界 1.4s
        # start = time.time()
        # binary_map = (protrusion_map > 0.00001).astype(np.float32)
        # cv2.imshow('binary map 1', to_normed_rgb(binary_map))
        # cv2.waitKey(-1)
        # end = time.time()
        # print(f"Binary map computation time: {end - start:.4f} seconds")

        # 获取深度图的点云
        start = time.time()
        s = depth2cloud(self.cam_matrix, depth)
        end = time.time()
        print(f"Point cloud computation time: {end - start:.4f} seconds")

        # 获取点云的光线方向
        start = time.time()
        optical_rays = normalize_vectors(s)
        end = time.time()
        print(f"Optical rays normalization time: {end - start:.4f} seconds")

        # 计算遮挡区域（即应该有阴影的区域）0.04s
        # start = time.time()
        # occluded_areas = self.calculate_occluded_areas(protrusion_map, optical_rays)
        # end = time.time()
        # print(f"Occluded areas computation time: {end - start:.4f} seconds")

        # **可选：弹性变形 (如果需要的话)
        # if self.apply_elastic_deformation:
        #     start = time.time()
        #     elastic_deformation = cv2.filter2D(self.bkg_depth - depth, -1, gkern2(55, 5))
        #     elastic_deformation = np.maximum((1 - occluded_areas) * elastic_deformation, np.zeros_like(occluded_areas))
        #     depth = np.minimum(depth, self.bkg_depth - elastic_deformation).astype(np.float32)
        #     end = time.time()
        #     print(f"Elastic deformation computation time: {end - start:.4f} seconds")

        # 计算表面法线和视线方向 0.1s
        start = time.time()
        n = -normals(s)
        v = -optical_rays
        end = time.time()
        print(f"Normals and optical rays computation time: {end - start:.4f} seconds")

        # 计算接触差异并将其归一化为百分比
        start = time.time()
        contact_diff = np.abs(protrusion_map)
        contact_diff = np.clip(contact_diff, 0, 0.03)
        contact_percentage = contact_diff / 0.03
        shadow_factor = np.clip(self.internal_shadow * contact_percentage, 0, 1)
        ambient_component = self.background_img * (self.ia - shadow_factor)[:, :, np.newaxis]
        end = time.time()
        print(f"Contact difference and shadow factor computation time: {end - start:.4f} seconds")

        # 计算光照 0.4s
        start = time.time()
        I = ambient_component + np.sum([self._spec_diff(lm, v, n, s) for lm in self.lights], axis=0)
        end = time.time()
        print(f"Illumination computation time: {end - start:.4f} seconds")

        # 处理图像和应用阴影
        start = time.time()
        I_rgb = (I * 255.0)
        I_rgb[I_rgb > 255.0] = 255.0
        I_rgb[I_rgb < 0.0] = 0.0
        I_rgb = I_rgb.astype(np.uint8)
        end = time.time()
        print(f"Image processing and shadow application time: {end - start:.4f} seconds")

        # ** 选择性：高斯模糊应用
        start = time.time()
        mask = n[:, :, 2] > -0.1
        I_rgb_new = I_rgb.copy()
        kernel_size = (5, 5)
        sigma = 5
        gaussian_kernel = cv2.getGaussianKernel(kernel_size[0], sigma)
        gaussian_kernel_2d = np.outer(gaussian_kernel, gaussian_kernel.T)
        gaussian_kernel_2d /= gaussian_kernel_2d.sum()
        for channel in range(I_rgb_new.shape[2]):
            I_rgb_new[:, :, channel] = np.where(mask, cv2.filter2D(I_rgb_new[:, :, channel], -1, gaussian_kernel_2d), I_rgb_new[:, :, channel])
        end = time.time()
        print(f"Gaussian blur application time: {end - start:.4f} seconds")

        # ** 选择性：绘制遮挡图和覆盖图像
        # start = time.time()
        # I_rgb_new = np.clip(I_rgb_new, 0, 255).astype(np.uint8)
        # occluded_map = (occluded_areas.astype(np.float32) * 255).astype(np.uint8)
        # occluded_map_3channel = cv2.cvtColor(occluded_map, cv2.COLOR_GRAY2BGR)
        # I_rgb_overlay = cv2.addWeighted(I_rgb, 0.5, occluded_map_3channel, 0.5, 0)
        # I_rgb_overlay = np.clip(I_rgb_overlay, 0, 255).astype(np.uint8)
        # cv2.imshow('Overlay Image', I_rgb_overlay)
        # end = time.time()
        # print(f"Overlay image computation time: {end - start:.4f} seconds")

        return I_rgb_new
