import cv2
import numpy as np
from pathlib import Path
import torch
from torch import Tensor
from PIL import Image


def find_folders(root: Path) -> list[Path]:
    subsubdirs = []
    for subdir in sorted(root.glob("*")):
        for subsubdir in sorted(subdir.glob("*")):
            subsubdirs.append(subsubdir)
    return subsubdirs


def load_xy(dir: Path) -> tuple[Tensor, Tensor]:
    """
    x ... input image; =AOS image; in Kelvin; HxW
    y ... GT image; =floor texture; in Kelvin; HxW
    """
    y = torch.from_numpy(np.array(Image.open(dir / "GT.tiff")))
    min_temp, max_temp = y.min(), y.max()
    x = torch.from_numpy(cv2.imread(
        str(dir / "integrall_0.png"), -1
    )[:, :, 0].astype(np.float32))  # in [0, 255]; pixel
    x = ((max_temp - min_temp) * (x / 255)) + min_temp  # in Kelvin
    return x, y


# def pix_to_temp(image_path: str, min_max_temp_path: str) -> np.ndarray:
#     """Convert pixels to temperatures for processing."""
#     image = cv2.imread(image_path, -1).astype(np.float32)  # HxW

#     with open(min_max_temp_path, "r") as fp:
#         min_temp, max_temp = map(float, fp.read().split(','))

#     return ((max_temp - min_temp) * (image / 255)) + min_temp


# def temp_to_pix(image: np.ndarray) -> np.ndarray:  # HxW
#     """Convert temperatures to pixels for visualization."""
#     min_temp, max_temp = image.min(), image.max()
#     return np.round(((image - min_temp) / (max_temp - min_temp)) * 255).astype(np.uint8)
