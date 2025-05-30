import cv2
import numpy as np
from pathlib import Path
import torch
from torch import Tensor
from PIL import Image
import re

from ..utils.image import tone_mapping


def find_folders(root: Path) -> list[Path]:
    subsubdirs = []
    for subdir in sorted(root.glob("*")):
        for subsubdir in sorted(subdir.glob("*")):
            subsubdirs.append(subsubdir)
    return subsubdirs


def read_min_max(file: Path):
    with open(file, "r") as fp:
        line = fp.read()
    min_temp, max_temp = [float(t) for t in line.split(",")]
    return min_temp, max_temp


def write_min_max(file: Path, mi, ma):
    with open(file, "w") as fp:
        fp.write(f"{mi},{ma}")


def load_single_view(image_file: Path, txt_file: Path):
    min_temp, max_temp = read_min_max(txt_file)
    x = torch.from_numpy(cv2.imread(str(image_file), -1).astype(np.float32))  # HxW
    x = ((max_temp - min_temp) * (x / 255)) + min_temp  # in Kelvin
    return x


def numericalSort(full_path: Path):
    file_name = full_path.stem  # no extention
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(file_name)
    parts[1::2] = map(int, parts[1::2])
    return parts


def load_center_view(folder: Path) -> Tensor:
    image_files = sorted((folder / "images").glob("*.png"), key=numericalSort)
    n_files = len(image_files)
    assert n_files % 2 == 1

    txt_files = sorted((folder / "images").glob("*.txt"), key=numericalSort)
    assert len(txt_files) == n_files

    idx = int(n_files / 2)
    x = load_single_view(image_files[idx], txt_files[idx])
    return x


def drone_flight_gif(
    folder: Path,
    min_temp: float, 
    max_temp: float,
    output_path: Path,
    frame_duration: int = 500,
):
    """Take global min./max. temp. from Ground Fire GT image."""
    image_files = sorted((folder / "images").glob("*.png"), key=numericalSort)
    txt_files = sorted((folder / "images").glob("*.txt"), key=numericalSort)

    frames = []
    for idx in range(len(image_files)):
        x = load_single_view(image_files[idx], txt_files[idx])[None, :, :]
        frames.append(tone_mapping(x, min_temp, max_temp, cv2.COLORMAP_INFERNO))

    # Save the first frame and append the rest as a GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,  # Duration of each frame in ms
        loop=0  # Infinite loop
    )
    print(f"GIF created successfully and saved to {output_path}")


def load_xy(dir: Path, normalized: bool = False) -> tuple[Tensor, Tensor]:
    """
    x ... input image; =AOS image; in Kelvin; HxW
    y ... GT image; =floor texture; in Kelvin; HxW
    """
    y = torch.from_numpy(np.array(Image.open(dir / "GT.tiff")))
    min_temp, max_temp = y.min(), y.max()
    if min_temp == max_temp:
        min_max_file = dir / "min_max_temp.txt"
        if min_max_file.exists():
            min_temp, max_temp = read_min_max(min_max_file)
        else:
            min_temps, max_temps = [], []
            files = (dir / "images").glob("*.txt")
            for file in files:
                min_temp, max_temp = read_min_max(file)
                min_temps.append(min_temp)
                max_temps.append(max_temp)

            # Find global min, max from individual views.
            min_temp = min(min_temps)
            max_temp = max(max_temps)
            # Store preprocessed min/max temperatures.
            write_min_max(min_max_file, min_temp, max_temp)

    if normalized:
        x = torch.from_numpy(cv2.imread(
            str(dir / "integrall_normalized_0.png"), -1
        )[:, :, 0].astype(np.float32))  # in [0, 255]; pixel
        min_temp, max_temp = read_min_max(dir / "global_min_max_temp.txt")
    else:
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
