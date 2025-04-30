import numpy as np
import cv2
import torch
import torch.nn.functional as F

import scipy.ndimage.filters as fi

""" linear algebra """


def dot_vectors(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.sum(a * b, dim=2)
    else:
        # 将 NumPy 数组转换为 PyTorch 张量
        a = torch.tensor(a, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        b = torch.tensor(b, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        return torch.sum(a * b, dim=2)


def normalize_vectors(m):
    if isinstance(m, torch.Tensor):
        norms = torch.norm(m, dim=2, keepdim=True)
        norms[norms == 0] = 1
        return m / norms
    else:
        m = torch.tensor(m, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        norms = torch.norm(m, dim=2, keepdim=True)
        norms[norms == 0] = 1
        return m / norms


def norm_vectors(m, zero=1.e-9):
    if isinstance(m, torch.Tensor):
        norms = torch.norm(m, dim=2)
        norms = torch.where(((-1 * zero) < norms) & (norms < zero), torch.ones_like(norms), norms)
        return norms
    else:
        m = torch.tensor(m, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        norms = torch.norm(m, dim=2)
        norms = torch.where(((-1 * zero) < norms) & (norms < zero), torch.ones_like(norms), norms)
        return norms



def proj_vectors(u, n):
    if isinstance(u, torch.Tensor):
        dot_product = torch.sum(u * n, dim=2, keepdim=True)
        return dot_product * normalize_vectors(n)
    else:
        return dot_vectors(u, n)[:, :, np.newaxis] * normalize_vectors(n)


def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen // 2, kernlen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)


def partial_derivative(mat, direction):
    assert direction in ('x', 'y'), "The derivative direction must be 'x' or 'y'"
    
    if direction == 'x':
        kernel = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    elif direction == 'y':
        kernel = torch.tensor([
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Ensure both mat and kernel are on the same device (GPU)
    device = mat.device
    kernel = kernel.repeat(3, 1, 1, 1)
    kernel = kernel.to(device)
    #print('kernel shape', kernel.shape)
    
    # Ensure mat has 4 dimensions (batch_size, channels, height, width)
    mat = mat.permute(2, 0, 1).unsqueeze(0).to(device)
    #print('mat shape:', mat.shape)
    
    # Perform the convolution
    res = F.conv2d(mat, kernel, padding=1, groups=3)
    #print('res shape', res.shape)
    
    # Squeeze the result to remove unnecessary dimensions
    return res.squeeze(0).permute(1, 2, 0)

def normals(s):
    device = s.device if isinstance(s, torch.Tensor) and s.is_cuda else torch.device('cpu')
    dx = normalize_vectors(partial_derivative(s, 'x'))
    dy = normalize_vectors(partial_derivative(s, 'y'))
    normal = torch.cross(dx, dy, dim=-1)
    normal = normalize_vectors(normal)
    return normal