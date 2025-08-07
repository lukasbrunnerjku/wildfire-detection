import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF 
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
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
        same: bool = True,
    ):
        super().__init__()
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.win = win
        self.K = K
        
        self.pad = None
        if same:
            p = int((win_size - 1) / 2)
            self.pad = (p, p, p, p)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        data_range: float = 255,
        size_average: bool = True,
        nonnegative_ssim: bool = False,
    ):
        if self.pad is not None:
            x = F.pad(x, self.pad, mode="constant", value=x.mean())
            y = F.pad(y, self.pad, mode="constant", value=y.mean())

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
        

@torch.no_grad()
def get_ssim(model: SSIM, aos: Tensor, gt: Tensor) -> Tensor:
    """More similar regions should be punished more if incorrect."""
    x = TF.normalize(aos, aos.mean(), aos.std())  # Bx1xHxW
    y = TF.normalize(gt, gt.mean(), gt.std())  # Bx1xHxW

    dxy = float(torch.max(x.max(), y.max())) - float(torch.min(x.min(), y.min()))

    ssim: Tensor = model(
        x, y,
        data_range=dxy,
        nonnegative_ssim=True,
        size_average=False
    )
    return ssim.clip(max=1.0)  # Bx1xHxW; in [0.0, 1.0]


def dilation(x: Tensor, k=3, s=1, p=1) -> Tensor:
    return F.max_pool2d(x, k, s, p)


def erosion(x: Tensor, k=3, s=1, p=1) -> Tensor:  # min pool
    return -F.max_pool2d(-x, k, s, p)


def morph(x: Tensor, n=3) -> Tensor:
    for _ in range(n):
        x = dilation(x)
    for _ in range(n):
        x = erosion(x)
    return x


def visualize(fp: str, aos: Tensor, gt: Tensor, hm: Tensor):
    from ..utils.image import tone_mapping
    
    _min, _max = float(gt.min()), float(gt.max())
    
    # NOTE: *_map are PIL Image objects!
    aos_map = tone_mapping(aos[None, ...], _min, _max)
    gt_map = tone_mapping(gt[None, ...], _min, _max)
    
    # 1 ... gt and aos have similar structure => apply correction
    mask = morph((hm > 0.6).to(dtype=torch.float32)[None, None, ...])[0, 0, ...]
    pred = (1.0 - mask) * aos + mask * gt
    pred_map = tone_mapping(pred[None, ...], _min, _max)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Plot RGB images
    axes[0].imshow(gt_map)
    axes[0].set_title("GT")
    axes[0].axis("off")
    
    axes[1].imshow(aos_map)
    axes[1].set_title("AOS")
    axes[1].axis("off")
    
    # Plot heatmap
    im = axes[2].imshow(hm, vmin=0.0, vmax=1.0)
    axes[2].set_title("Heatmap")
    axes[2].axis("off")  # remove ticks
    # Add colorbar for heatmap
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    axes[3].imshow(mask, vmin=0.0, vmax=1.0, cmap="gray")
    axes[3].set_title("Mask")
    axes[3].axis("off")
    
    axes[4].imshow(pred_map)
    axes[4].set_title("Pred")
    axes[4].axis("off")
    
    plt.tight_layout(pad=1.1)
    plt.savefig(fp)
    plt.close()
    

if __name__ == "__main__":
    from .utils import load_xy
    
    dir = "/mnt/data/wildfire/IR/root/Batch-1/0/101186b2bd74421681d0958ea9db264c"
    
    aos, gt = load_xy(Path(dir))  # HxW
    model = SSIM()

    hm = get_ssim(model, aos[None, None, ...], gt[None, None, ...])[0, 0, ...]  # HxW
    
    visualize("/mnt/data/wildfire/IR/test.png", aos, gt, hm)
    