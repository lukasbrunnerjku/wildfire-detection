import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF 
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from .utils import load_xy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from typing import Optional
import warnings


def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: Tensor, win: Tensor) -> Tensor:
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(
    x: Tensor,
    y: Tensor,
    data_range: float,
    win: Tensor,
    K: tuple[float, float] = (0.01, 0.03)
) -> Tensor:
    r""" Calculate ssim index for x and y

    Args:
        x (torch.Tensor): images
        y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: ssim result.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = x.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(x.device, dtype=x.dtype)

    mu1 = gaussian_filter(x, win)
    mu2 = gaussian_filter(y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(x * x, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(y * y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(x * y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    return ssim_map


def ssim_torch(
    x: Tensor,
    y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: tuple[float, float] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
) -> Tensor:
    r"""
    Args:
        x (torch.Tensor): a batch of images, (N,C,H,W)
        y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not x.shape == y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {x.shape} and {y.shape}.")

    for d in range(len(x.shape) - 1, 1, -1):
        x = x.squeeze(dim=d)
        y = y.squeeze(dim=d)

    if len(x.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {x.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([x.shape[1]] + [1] * (len(x.shape) - 1))

    ssim_map = _ssim(x, y, data_range=data_range, win=win, K=K)

    if nonnegative_ssim:
        ssim_map = torch.relu(ssim_map)

    if size_average:
        return ssim_map.mean((2, 3))  # BxC
    else:
        return ssim_map  # BxCxHxW
    

class SSIM(nn.Module):

    def __init__(
        self,
        win_size: int = 11,
        win_sigma: float = 1.5,
        win: Optional[Tensor] = None,
        K: tuple[float, float] = (0.01, 0.03),
    ):
        super().__init__()
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.win = win
        self.K = K

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        data_range: float = 255,
        size_average: bool = True,
        nonnegative_ssim: bool = False,
    ):
        return ssim_torch(
            x,
            y,
            data_range=data_range,
            size_average=size_average,
            win_size=self.win_size,
            win_sigma=self.win_sigma,
            win=self.win,
            K=self.K,
            nonnegative_ssim=nonnegative_ssim,
        )


if __name__ == "__main__":
    dir = "/mnt/data/wildfire/IR/Batch-1/0/101186b2bd74421681d0958ea9db264c"
    x0, y0 = load_xy(Path(dir))

    x = TF.normalize(x0[None,...], x0.mean(), x0.std())[0]  # HxW
    y = TF.normalize(y0[None,...], y0.mean(), y0.std())[0]  # HxW

    # Difference between maximum and minimum possible values
    gmax = float(max(x.max(), y.max()))
    gmin = float(min(x.min(), y.min()))
    dxy = gmax - gmin

    print(x.mean(), x.std(), x.min(), x.max())
    print(y.mean(), y.std(), y.min(), y.max())
    print(dxy)

    hm_torch = ssim_torch(
        x[None, None, ...], y[None, None, ...],
        data_range=dxy, size_average=False,
        win_size=7, win_sigma=1.5,
        nonnegative_ssim=True,
    )[0, 0, ...] # HxW; [0, 1.0]
    
    x, y = x.numpy(), y.numpy()

    _, hm = ssim(x, y, data_range=dxy, full=True, win_size=7)  # HxW; [-inf, 1.0]
    hm = hm.clip(0, 1)

    ax = plt.subplot(1, 2, 1)
    ax.imshow(hm_torch, vmin=0, vmax=1)
    ax = plt.subplot(1, 2, 2)
    ax.imshow(hm, vmin=0, vmax=1)
    plt.savefig("test.png")

    mask = hm > 0.8

    plt.clf()
    ax = plt.subplot(1, 3, 2)
    ax.imshow(mask, vmin=0, vmax=1)

    x[~mask] = gmin
    ax = plt.subplot(1, 3, 1)
    ax.imshow(x, vmin=gmin, vmax=gmax)

    y[~mask] = gmin
    ax = plt.subplot(1, 3, 3)
    ax.imshow(y, vmin=gmin, vmax=gmax)

    plt.savefig("test1.png")

    z = y0 - x0
    z = z.numpy()

    dT_Kelvin = 5.0

    plt.clf()
    ax = plt.subplot(1, 1, 1)
    ax.imshow(z, vmin=-dT_Kelvin, vmax=dT_Kelvin, cmap="bwr")
    plt.savefig("test2.png")

    z[~mask] = 0
    plt.clf()
    ax = plt.subplot(1, 1, 1)
    ax.imshow(z, vmin=-dT_Kelvin, vmax=dT_Kelvin, cmap="bwr")
    plt.savefig("test3.png")

    """
    We want to measure the correct temperature, we want to decrease temperature
    deviations introduced by AOS, can we measure unseen parts? No. Can we measure
    partially observed parts that at least kept some of their structural appearance?
    yes.

    => Allow change only in reasonably structurally similar areas. Penalize via loss
    objective only in those areas as well. Modelling via residuals. Utilize similarity
    map as input to the correction algorithm, as done in inpainting.

    Should we let a correction algorithm alter the structure fundamentally? No, not
    if it has not been provided more than just the AOS image. No proper structural
    change could be causaly explained. Keep in mind we want to minimize halucinations.

    => Introduce auxilary loss based on ssim but thresholded to not penalize above ie.
    a threshold of 0.8 (as small changes could be reasonable).
    """
