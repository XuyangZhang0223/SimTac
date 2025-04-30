#!/usr/bin/env python
import cv2
import numpy as np
import time
import open3d as o3d
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from .utils.camera_gpu import get_camera_matrix, depth2cloud
from .utils.maths_gpu import normalize_vectors, gkern2, dot_vectors, normals, proj_vectors, partial_derivative
from .utils.vis_img import to_normed_rgb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_gaussian_kernel_4d(kernel_size=5, sigma=5, device='cuda'):
    """Create a 4D Gaussian kernel for convolution directly on GPU."""
    # Create a 1D Gaussian kernel on GPU
    x = torch.arange(kernel_size, dtype=torch.float32, device=device)
    x = x - (kernel_size - 1) / 2
    gaussian_kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    gaussian_kernel_1d /= gaussian_kernel_1d.sum()

    # Create a 2D Gaussian kernel
    gaussian_kernel_2d = gaussian_kernel_1d.unsqueeze(1) @ gaussian_kernel_1d.unsqueeze(0)
    gaussian_kernel_2d /= gaussian_kernel_2d.sum()

    # Expand to 4D for convolution
    gaussian_kernel_4d = gaussian_kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size, kernel_size)
    gaussian_kernel_4d = gaussian_kernel_4d.expand(3, 1, kernel_size, kernel_size)  # (3, 1, kernel_size, kernel_size)

    return gaussian_kernel_4d

class SimulationModel:

    def __init__(self, **config):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.default_ks = 0.15
            self.default_kd = 0.5
            self.default_alpha = 100
            self.ia = config['ia'] or 1
            self.fov = config['fov'] or 100

            self.lights = config['light_sources']              
            self.rectify_fields = config['rectify_fields']

            self.bkg_depth = torch.tensor(config['background_depth'], dtype=torch.float32, device=device)
            self.cam_matrix = get_camera_matrix(self.bkg_depth.shape[::-1], self.fov)

            self.background_img = torch.tensor(config['background_img'], dtype=torch.float32, device=device)
            self.s_ref = depth2cloud(self.cam_matrix, self.bkg_depth).to(device)
            self.s_ref_n = normals(self.s_ref).to(device)

            self.apply_elastic_deformation = config.get('elastic_deformation', False)
            self.internal_shadow = config.get('internal_shadow', 0.15)

            self.ambient = self.background_img

            for light in self.lights:
                light['ks'] = light.get('ks', self.default_ks)
                light['kd'] = light.get('kd', self.default_kd)
                light['alpha'] = light.get('alpha', self.default_alpha)
                light['field'] = torch.tensor(light['field'], dtype=torch.float32, device=device)
                light['color_map'] = torch.tensor(np.tile(np.array(np.array(light['color']) / 255.0)
                                                        .reshape((1, 1, 3)), self.s_ref.shape[0:2] + (1,)),
                                                dtype=torch.float32, device=device)

            self.texture_sigma = config.get('texture_sigma', 0.00001)
            self.t = config.get('t', 3)
            self.sigma = config.get('sigma', 7)
            self.kernel_size = config.get('kernel_size', 21)

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
        n = n.to(self.device)
        v = v.to(self.device)
        lm = - lm_data['field']
        color = lm_data['color_map']

        if self.rectify_fields:
            lm = normalize_vectors(lm - proj_vectors(lm, self.s_ref_n))

        # Shared calculations
        lm_n = dot_vectors(lm, n).clamp(min=0.0)
        Rm = 2.0 * lm_n.unsqueeze(-1) * n - lm  

        # diffuse component
        diffuse_l = lm_n * imd

        # specular component
        spec_l = (dot_vectors(Rm, v) ** alpha) * ims

        return (diffuse_l + spec_l).unsqueeze(-1) * color



    def calculate_occluded_areas(self, protrusion_map, optical_rays):
        binary_map = (protrusion_map > 0.00001).float()

        areas_x = partial_derivative(binary_map, 'x')
        areas_y = partial_derivative(binary_map, 'y')

        sign_comparison = torch.equal(torch.sign(optical_rays[:, :, :2]), torch.sign(torch.stack([areas_x, areas_y], dim=-1)))

        occluded_areas = torch.clip(sign_comparison.sum(dim=-1) / 0.05, 0, 1)

        kernel = torch.ones((3, 3), dtype=torch.uint8, device=device)
        occluded_areas = F.conv2d(occluded_areas.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze()

        occluded_areas = F.conv2d(occluded_areas.unsqueeze(0).unsqueeze(0), gkern2(55, 5).unsqueeze(0).unsqueeze(0), padding=27).squeeze()

        occluded_areas = (occluded_areas - occluded_areas.min()) / (occluded_areas.max() - occluded_areas.min())
        occluded_areas *= (1 - binary_map)

        return occluded_areas

    def calculate_occluded_areas_alternative(self, surface_normals, optical_rays, threshold=0.95):
        dot_product = torch.abs(torch.sum(surface_normals * optical_rays, dim=-1))
        occlusion_map = (dot_product > threshold).float()

        kernel = torch.ones((3, 3), dtype=torch.uint8, device=device)
        occlusion_map = F.conv2d(occlusion_map.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze()

        occlusion_map = F.gaussian_blur(occlusion_map.unsqueeze(0), (55, 55)).squeeze()

        occlusion_map = (occlusion_map - occlusion_map.min()) / (occlusion_map.max() - occlusion_map.min())

        return occlusion_map

    def generate(self, depth):

        start1 = time.time()

        # 计算 protrusion_map, 计算深度差的图像
        start = time.time()
        #depth = depth.clone().detach().to(torch.float32).to(self.device)
        depth = torch.tensor(depth, dtype=torch.float32, device=device)
        protrusion_map = self.bkg_depth - depth
        end = time.time()
        #print(f"Protrusion map computation time: {end - start:.4f} seconds")

        # 计算二值图像
        #start = time.time()
        #binary_map = (protrusion_map > 0).float()
        #binary_map = (protrusion_map != 0).float() # this mask can be used to clip the rendered image
        # cv2.imshow('binary map', to_normed_rgb(binary_map.cpu().numpy()))
        # cv2.waitKey(-1)
        #kernel = torch.ones((9, 9), dtype=torch.float32, device=device)
        #binary_map_1 = F.conv2d(binary_map.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding = (kernel.size(0) - 1) // 2).squeeze()
        #binary_map = (binary_map_1 == kernel.sum()).float()
        # cv2.imshow('binary map 1', to_normed_rgb(binary_map.cpu().numpy()))
        # cv2.waitKey(-1)
        #end = time.time()
        #print(f"Binary map computation time: {end - start:.4f} seconds")

        # 获取深度图的点云
        start = time.time()
        s = depth2cloud(self.cam_matrix, depth).to(self.device)
        end = time.time()
        #print(f"Point cloud computation time: {end - start:.4f} seconds")

        # 获取点云的光线方向
        start = time.time()
        optical_rays = normalize_vectors(s)
        end = time.time()
        #print(f"Optical rays normalization time: {end - start:.4f} seconds")

        # 计算遮挡区域
        #start = time.time()
        #occluded_areas = self.calculate_occluded_areas(protrusion_map, optical_rays)
        #end = time.time()
        #print(f"Occluded areas computation time: {end - start:.4f} seconds")

        # **可选：弹性变形
        # if self.apply_elastic_deformation:
        #     start = time.time()
        #     elastic_deformation = F.conv2d(self.bkg_depth - torch.tensor(depth, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0), gkern2(55, 5).unsqueeze(0).unsqueeze(0), padding=27).squeeze()
        #     elastic_deformation = torch.maximum((1 - occluded_areas) * elastic_deformation, torch.zeros_like(occluded_areas))
        #     depth = torch.minimum(torch.tensor(depth, dtype=torch.float32, device=device), self.bkg_depth - elastic_deformation).cpu().numpy()
        #     end = time.time()
        #     print(f"Elastic deformation computation time: {end - start:.4f} seconds")

        # 计算表面法线和视线方向
        start = time.time()
        n = -normals(s)
        v = -optical_rays
        end = time.time()
        #print(f"Normals and optical rays computation time: {end - start:.4f} seconds")

        # 计算接触差异并将其归一化为百分比
        start = time.time()
        contact_diff = torch.abs(protrusion_map)
        contact_diff = torch.clip(contact_diff, 0, 0.03)
        contact_percentage = contact_diff / 0.03
        shadow_factor = torch.clip(self.internal_shadow * contact_percentage, 0, 1)
        ambient_component = self.background_img * (self.ia - shadow_factor)[:, :, np.newaxis]
        end = time.time()
        #print(f"Contact difference and shadow factor computation time: {end - start:.4f} seconds")

        # 计算光照
        start = time.time()
        I = ambient_component + torch.sum(torch.stack([self._spec_diff(lm, v, n, s) for lm in self.lights], dim=0), dim=0)
        end = time.time()
        #print(f"Illumination computation time: {end - start:.4f} seconds")

        # 处理图像和应用阴影
        start = time.time()
        I_rgb = (I * 255.0).clamp(0, 255).byte()
        end = time.time()
        #print(f"Image processing and shadow application time: {end - start:.4f} seconds")

        # ** 选择性：高斯模糊应用
        # start = time.time()
        # mask = n[:, :, 2] > -0.1
        # I_rgb_new = I_rgb.clone()
        # I_rgb_new = I_rgb_new.cpu().numpy()
        # kernel_size = (5, 5)
        # sigma = 5
        # gaussian_kernel = cv2.getGaussianKernel(kernel_size[0], sigma)
        # gaussian_kernel_2d = np.outer(gaussian_kernel, gaussian_kernel.T)
        # gaussian_kernel_2d /= gaussian_kernel_2d.sum()
        # for channel in range(I_rgb_new.shape[2]):
        #     I_rgb_new[:, :, channel] = np.where(mask.cpu().numpy(), cv2.filter2D(I_rgb_new[:, :, channel], -1, gaussian_kernel_2d), I_rgb_new[:, :, channel])
        # end = time.time()
        # print(f"Gaussian blur application time: {end - start:.4f} seconds")


        # Gaussian blur for shadow area
        start_total = time.time()   
        I_rgb = I_rgb.clone().detach().to(torch.float32).to(device)
        n = n.clone().detach().to(device)
        mask = n[:, :, 2] > -0.1
        kernel_size = 33
        kernel_size_mask = 33
        sigma = 5
        gaussian_kernel_4d = create_gaussian_kernel_4d(kernel_size, sigma, device)
        gaussian_kernel_4d_mask = create_gaussian_kernel_4d(kernel_size_mask, sigma, device)
        # Start timer for Gaussian blur application
        I_rgb = I_rgb.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        mask = mask.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        mask = mask.repeat(1, 3, 1, 1)  # (1, 3, H, W)
        I_rgb_blurred = F.conv2d(I_rgb, gaussian_kernel_4d, padding=kernel_size // 2, groups=3)
        mask = F.conv2d(mask, gaussian_kernel_4d_mask, padding=kernel_size_mask // 2, groups=3)
        # Use mask to compute the new image efficiently
        I_rgb_new = I_rgb_blurred * mask*(7/10) + I_rgb * (1 - mask)   
        # Convert to numpy array and move to CPU in one go

        ## need to confirm whether the transfer from tensor to numpy is needed
        I_rgb_new = I_rgb_new.squeeze(0).permute(1, 2, 0).cpu().numpy()
        end_total = time.time()
        #print(f"Gaussian blur time: {end_total - start_total:.4f} seconds")
        

        # ** 选择性：绘制遮挡图和覆盖图像
        #start = time.time()
        #I_rgb_new = I_rgb_new.cpu().numpy()
        #occluded_map = (occluded_areas.cpu().numpy() * 255).astype(np.uint8)
        #occluded_map_3channel = cv2.cvtColor(occluded_map, cv2.COLOR_GRAY2BGR)
        #I_rgb_overlay = cv2.addWeighted(I_rgb_new, 0.5, occluded_map_3channel, 0.5, 0)
        #I_rgb_overlay = np.clip(I_rgb_overlay, 0, 255).astype(np.uint8)
        #cv2.imshow('Overlay Image', I_rgb_overlay)
        #end = time.time()
        #print(f"Overlay image computation time: {end - start:.4f} seconds")

        end1 = time.time()
        #print(f"total time: {end1 - start1:.4f} seconds")

        return I_rgb_new
    

    def generate_bkg_image(self, depth):

        depth = torch.tensor(depth, dtype=torch.float32, device=device)

        # 获取深度图的点云
        start = time.time()
        s = depth2cloud(self.cam_matrix, depth).to(self.device)
        end = time.time()
        print(f"Point cloud computation time: {end - start:.4f} seconds")

        # 获取点云的光线方向
        start = time.time()
        optical_rays = normalize_vectors(s)
        end = time.time()
        print(f"Optical rays normalization time: {end - start:.4f} seconds")

        # 计算表面法线和视线方向
        start = time.time()
        n = -normals(s)
        v = -optical_rays
        end = time.time()
        print(f"Normals and optical rays computation time: {end - start:.4f} seconds")

        # 计算光照
        start = time.time()
        #I = ambient_component + torch.sum(torch.stack([self._spec_diff(lm, v, n, s) for lm in self.lights], dim=0), dim=0)
        I = torch.sum(torch.stack([self._spec_diff(lm, v, n, s) for lm in self.lights], dim=0), dim=0)
        end = time.time()
        print(f"Illumination computation time: {end - start:.4f} seconds")

        # 处理图像和应用阴影
        start = time.time()
        I_rgb = (I * 255.0).clamp(0, 255).byte()
        end = time.time()
        print(f"Image processing and shadow application time: {end - start:.4f} seconds")
        I_rgb = I_rgb.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)


        ## need to confirm whether the transfer from tensor to numpy is needed
        I_rgb_new = I_rgb.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()


        return I_rgb_new
