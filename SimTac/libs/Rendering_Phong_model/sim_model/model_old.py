
#!/usr/bin/env python
import cv2
import numpy as np

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
        # Calculate the protrusion_map, image with values of depth difference 
        protrusion_map = self.bkg_depth - depth

        # calculate the binary map, this is the outer boundary of the contact area
        binary_map = (protrusion_map > 0.00001).astype(np.float32)
        cv2.imshow('binary map 1', to_normed_rgb(binary_map))
        cv2.waitKey(-1)

        # binary_map_2 = (protrusion_map > 0.0008).astype(np.float32)
        # cv2.imshow('binary map 2', to_normed_rgb(binary_map_2))
        # cv2.waitKey(-1)
        

        # get surface point-cloud from depth map
        s = depth2cloud(self.cam_matrix, depth)

        # s is the point cloud of the contact surface, we can get the vecter of each point by using the normalize_vectors function
        optical_rays = normalize_vectors(s)

        # Calculate the occluded areas, areas that should have shadow (That should be modified)
        occluded_areas = self.calculate_occluded_areas(protrusion_map, optical_rays)
        # print('areas', np.min(occluded_areas), np.max(occluded_areas))
        # cv2.imshow('areas', to_normed_rgb(occluded_areas))
        # cv2.waitKey(-1)


        # # **elastic deformation (as in the paper, submitted to RSS)
        # if self.apply_elastic_deformation:
        #     elastic_deformation = cv2.filter2D(self.bkg_depth - depth, -1, gkern2(55, 5))

        #     elastic_deformation = np.maximum((1 - occluded_areas) * elastic_deformation, np.zeros_like(occluded_areas))
        #     depth = np.minimum(depth, self.bkg_depth - elastic_deformation).astype(np.float32)
        #     print(depth.dtype)


        # Optical Rays = s - 0
        optical_rays = normalize_vectors(s)

        # illumination vectors (n, v) calculations
        n = - normals(s)
        v = - optical_rays
        
        #print('n', n)

        # Calculate the absolute difference between bkg_depth and depth using the precomputed protrusion_map
        contact_diff = np.abs(protrusion_map)

        # Clip the difference to a maximum value (e.g., 0.03)
        contact_diff = np.clip(contact_diff, 0, 0.03)

        # Normalize the difference to a percentage
        contact_percentage = contact_diff / 0.03

        # Multiply the internal_shadow by the percentage and clip it to a valid range (0 to 1)
        shadow_factor = np.clip(self.internal_shadow * contact_percentage, 0, 1)

        ambient_component = self.background_img * (self.ia - shadow_factor)[:, :, np.newaxis]

        I = ambient_component + np.sum([self._spec_diff(lm, v, n, s) for lm in self.lights], axis=0)

        I_rgb = (I * 255.0)
        #print('I_rgb', I_rgb)
        I_rgb[I_rgb > 255.0] = 255.0
        I_rgb[I_rgb < 0.0] = 0.0
        I_rgb = I_rgb.astype(np.uint8)
        


        # ** Luminance-based method to add shadow
        # 创建一个与 I_rgb 相同大小的深拷贝以存储新的像素值
        # I_rgb_new = I_rgb.copy()
        # darken_factor = 0.85
        # 将满足 n[:, :, 2] > 0 的像素值乘以相应的变暗系数
        # mask = n[:, :, 2] > 0
        # for i in range(3):  # 遍历每个颜色通道
        #     r = I_rgb[mask, 0]
        #     g = I_rgb[mask, 1]
        #     b = I_rgb[mask, 2]
        #     adjusted_r, adjusted_g, adjusted_b = adjust_luminance(r, g, b, darken_factor)
        #     I_rgb_new[mask, 0] = np.clip(adjusted_r, 0, 255).astype(np.uint8)
        #     I_rgb_new[mask, 1] = np.clip(adjusted_g, 0, 255).astype(np.uint8)
        #     I_rgb_new[mask, 2] = np.clip(adjusted_b, 0, 255).astype(np.uint8)


        ## ** visualization of the occluded_areas and the calculated mask (acoording to the gradient of the depth map)
        # mask = n[:, :, 2] > -0.001
        # I_rgb_float = I_rgb.astype(np.float32)
        # # 创建一个与 I_rgb 相同形状的全零数组
        # output = np.zeros_like(I_rgb_float)
        # # 将满足 mask 条件的区域设为 1（我们可以设置 RGB 三个通道为 1）
        # output[mask] = 1.0
        # # 将结果转换回 uint8 类型以便可视化
        # occluded_areas[mask] = 1.0
        # occluded_areas = (occluded_areas * 255).astype(np.uint8)
        # cv2.imshow('areas', to_normed_rgb(occluded_areas))
        # cv2.waitKey(-1)


        # # **Replace the covered area with soft shadow bkg
        # mask = n[:, :, 2] > -0.001
        # kernel = np.ones((3,3), np.uint8)
        # mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(np.bool_)
        # I_rgb_new = I_rgb.copy()
        # darkened_background = (self.background_img.astype(np.float32) *255)
        # #bkg = apply_soft_shadow(I_rgb_new*0.9, 0.7)
        # I_rgb_new[mask] = I_rgb_new[mask]*0+darkened_background[mask]*0.85


        # **Gaussian blury method to the mask area
        mask = n[:, :, 2] > -0.1
        I_rgb_new = I_rgb.copy()
        #kernel = np.ones((3, 3), np.uint8)
        #I_rgb_new = cv2.dilate(I_rgb_new, kernel, iterations=1)
        # I_rgb_new = cv2.GaussianBlur(I_rgb_new, (5, 5), 1)
        kernel_size = (5, 5)
        sigma = 5
        # Generate Gaussian kernel
        gaussian_kernel = cv2.getGaussianKernel(kernel_size[0], sigma)
        gaussian_kernel_2d = np.outer(gaussian_kernel, gaussian_kernel.T)
        # Normalize the kernel
        gaussian_kernel_2d /= gaussian_kernel_2d.sum()
        # Apply Gaussian blur to each channel of I_rgb
        for channel in range(I_rgb_new.shape[2]):
            I_rgb_new[:,:,channel] = np.where(mask, cv2.filter2D(I_rgb_new[:,:,channel], -1, gaussian_kernel_2d), I_rgb_new[:,:,channel])




        # print('max n[:, :, 2]', np.max(n[:, :, 2]))
        # print('min n[:, :, 2]', np.min(n[:, :, 2]))
        # mask_zero = n[:, :, 2] > -0.15
        # mask_zero = mask_zero.astype(np.uint8) * 255
        # kernel = np.ones((5, 5), np.uint8)
        # mask_zero = cv2.erode(mask_zero, kernel, iterations=1)
        # mask_zero = cv2.dilate(mask_zero, kernel, iterations=1)
        # #binary_map = cv2.GaussianBlur(binary_map, (5, 5), 1)
        # binary_map[mask_zero] = 0.0
        # binary_map = (binary_map * 255).astype(np.uint8)
        # cv2.imshow('mask map', binary_map)
        # cv2.waitKey(-1)






        # ** ORIGINAL CODE
        # # Convert to uint8
        I_rgb_new = np.clip(I_rgb_new, 0, 255).astype(np.uint8)
        
        # Normalize the occluded_mask values to 0 to 255
        occluded_map = (occluded_areas.astype(np.float32) * 255).astype(np.uint8)

        # Convert occluded_map to 3-channel image
        occluded_map_3channel = cv2.cvtColor(occluded_map, cv2.COLOR_GRAY2BGR)

        # Overlay the occluded_map over I_rgb with 50% opacity
        I_rgb_overlay = cv2.addWeighted(I_rgb, 0.5, occluded_map_3channel, 0.5, 0)

        # Normalize the RGB values
        I_rgb_overlay = np.clip(I_rgb_overlay, 0, 255).astype(np.uint8)



        # # **Visualization of binary map 2 
        # I_rgb_new = np.clip(I_rgb_new, 0, 255).astype(np.uint8)
        
        # # Normalize the occluded_mask values to 0 to 255
        # occluded_map = (binary_map_2.astype(np.float32) * 255).astype(np.uint8)

        # # Convert occluded_map to 3-channel image
        # occluded_map_3channel = cv2.cvtColor(occluded_map, cv2.COLOR_GRAY2BGR)

        # # Overlay the occluded_map over I_rgb with 50% opacity
        # I_rgb_overlay = cv2.addWeighted(I_rgb, 0.9, occluded_map_3channel, 0.1, 0)

        # # Normalize the RGB values
        # I_rgb_overlay = np.clip(I_rgb_overlay, 0, 255).astype(np.uint8)


        # ## **add shadow to the occluded areas
        # # 反转 occluded_areas 的值并将其转换为 float32 类型
        # occluded_map = (1 - occluded_areas).astype(np.float32)
        # # 将范围从 [0, 1] 映射到 [0, 0.5]
        # occluded_map = occluded_map * 0.5
        # # 将 occluded_map 转换为 [0, 255] 范围
        # occluded_map = (occluded_map * 255).astype(np.uint8)
        # # 将 occluded_map 转换为三通道图像
        # occluded_map_3channel = cv2.cvtColor(occluded_map, cv2.COLOR_GRAY2BGR)
        # # 创建掩码，标识出 occluded_map 中值小于 0.8 * 255 的部分
        # mask = occluded_map < 0.8 * 255
        # # 创建一个全零的图像，与 I_rgb 大小相同
        # I_rgb_overlay = np.zeros_like(I_rgb)
        # # 对掩码部分应用叠加处理
        # I_rgb_overlay[mask] = cv2.addWeighted(I_rgb[mask], 0.5, occluded_map_3channel[mask], 0.5, 0)
        # # 对未被掩码覆盖的部分，保留原始图像
        # I_rgb_overlay[~mask] = I_rgb[~mask]
        # # 归一化 RGB 值，确保在0到255范围内
        # I_rgb_overlay = np.clip(I_rgb_overlay, 0, 255).astype(np.uint8)


        # Display the occluded_map and the overlay image
        # cv2.imshow('Occluded Map', occluded_map)
        cv2.imshow('Overlay Image', I_rgb_overlay)
        # #cv2.imshow('Image', I_rgb)
        # cv2.waitKey(-1)

        return I_rgb_new