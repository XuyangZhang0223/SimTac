#!/usr/bin/env python
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.ndimage.filters as fi

light_sources = [
        {'position': [0, 1, 0.25], 'color': (255, 255, 255), 'kd': 0.6, 'ks': 0.5},  # white, top
        {'position': [-1, 0, 0.25], 'color': (255, 130, 115), 'kd': 0.5, 'ks': 0.3},  # blue, right
        {'position': [0, -1, 0.25], 'color': (108, 82, 255), 'kd': 0.6, 'ks': 0.4},  # red, bottom
        {'position': [1, 0, 0.25], 'color': (120, 255, 153), 'kd': 0.1, 'ks': 0.1},  # green, left
    ]
    
background = cv2.imread('background.png')
px2m_ratio = 5.4347826087e-05

elastomer_thickness = 0.004
min_depth = -0.01 # distance from the image sensor to the rigid glass outer surface unit: m
max_depth = 0.07

ka = 0.8
default_alpha = 5

t = 3
sigma = 7
kernel_size = 21

def gaus_noise(image, sigma):
    row, col = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy

def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen // 2, kernlen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)

def derivative(mat, direction):
    assert (direction == 'x' or direction == 'y'), "The derivative direction must be 'x' or 'y'"
    kernel = None
    if direction == 'x':
        kernel = [[-1.0, 0.0, 1.0]]
    elif direction == 'y':
        kernel = [[-1.0], [0.0], [1.0]]
    kernel = np.array(kernel, dtype=np.float64)
    return cv2.filter2D(mat, -1, kernel) / 2.0

def tangent(mat):
    dx = derivative(mat, 'x')
    dy = derivative(mat, 'y')
    img_shape = np.shape(mat)
    _1 = np.repeat([1.0], img_shape[0] * img_shape[1]).reshape(img_shape).astype(dx.dtype)
    unormalized = cv2.merge((-dx, -dy, _1))
    norms = np.linalg.norm(unormalized, axis=2)
    return (unormalized / np.repeat(norms[:, :, np.newaxis], 3, axis=2))

def solid_color_img(color, size):
    image = np.zeros(size + (3,), np.float64)
    image[:] = color
    return image

def add_overlay(rgb, alpha, color):
    s = np.shape(alpha)

    opacity3 = np.repeat(alpha, 3).reshape((s[0], s[1], 3))  # * 10.0

    overlay = solid_color_img(color, s)

    foreground = opacity3 * overlay
    background = (1.0 - opacity3) * rgb.astype(np.float64)
    res = background + foreground

    res[res > 255.0] = 255.0
    res[res < 0.0] = 0.0
    res = res.astype(np.uint8)

    return res

def segments(depth_map):
    case_depth = 20
    not_in_touch = np.copy(depth_map)

    not_in_touch[not_in_touch < max_depth] = 0.0
    not_in_touch[not_in_touch >= max_depth] = 1.0

    in_touch = 1 - not_in_touch

    return not_in_touch, in_touch

def protrusion_map(original, not_in_touch):
    protrusion_map = np.copy(original)
    protrusion_map[not_in_touch >= max_depth] = max_depth
    return protrusion_map


def internal_shadow(elastomer_depth):
    elastomer_depth_inv = max_depth - elastomer_depth
    elastomer_depth_inv = np.interp(elastomer_depth_inv, (0, elastomer_thickness), (0.0, 1.0))
    return elastomer_depth_inv

def phong_illumination(T, source_dir, kd, ks, alpha):
    dot = np.dot(T, np.array(source_dir)).astype(np.float64)
    difuse_l = dot * kd
    difuse_l[difuse_l < 0] = 0.0

    dot3 = np.repeat(dot[:, :, np.newaxis], 3, axis=2)

    R = 2.0 * dot3 * T - source_dir
    V = [0.0, 0.0, 1.0]

    spec_l = np.power(np.dot(R, V), alpha) * ks
    return difuse_l + spec_l

def generate(obj_depth):
    not_in_touch, in_touch = segments(obj_depth)
    protrusion_depth = protrusion_map(obj_depth, not_in_touch)
    elastomer_depth = protrusion_depth

    textured_elastomer_depth = gaus_noise(elastomer_depth, 0.000002)

    out = np.zeros((480, 640, 3), dtype=np.float64)
    #out = add_overlay(out, internal_shadow(protrusion_depth), (0.0, 0.0, 0.0))

    T = tangent(textured_elastomer_depth / px2m_ratio)

    for light in light_sources:
        ks = light['ks'] if 'ks' in light else default_ks
        kd = light['kd'] if 'kd' in light else default_kd
        alpha = light['alpha'] if 'alpha' in light else default_alpha
        
        out = add_overlay(out, phong_illumination(T, light['position'], kd, ks, alpha), light['color'])

    return out
