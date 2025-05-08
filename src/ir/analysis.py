import json
import torch
from torch import Tensor
import cv2
from tqdm import tqdm
from multiprocessing import Pool, Manager, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from functools import partial
from PIL import Image

from .utils import load_xy, find_folders
from ..utils.image import tone_mapping, pil_make_grid


def make_grid(x: Tensor, y: Tensor, z: Tensor, dt, common_min_max) -> Image.Image:
    if common_min_max:
        min_t = min(x.min(), y.min())
        max_t = max(x.max(), y.max())
        imgs = [
            tone_mapping(x[None, :, :], min=min_t, max=max_t),
            tone_mapping(y[None, :, :], min=min_t, max=max_t),
        ]
    else:
        imgs = [
            tone_mapping(x[None, :, :], min=x.min(), max=x.max()),
            tone_mapping(y[None, :, :], min=y.min(), max=y.max()),
        ]

    imgs.append(tone_mapping(z[None, :, :], min=-dt, max=dt, colormap=cv2.COLORMAP_JET))

    grid = pil_make_grid(imgs, ncol=len(imgs))
    return grid


def process_folder(index_and_subsubdir, dt, common_min_max, grid_folder, files_list, skipped_list):
    """Worker function to process each folder."""
    index, subsubdir = index_and_subsubdir   # "imap_unordered" supports only a single argument

    x, y = load_xy(subsubdir)
    z = y - x  # Residual target

    if torch.abs(z).mean() < 1.0:
        skipped_list.append(str(subsubdir))
        return

    grid = make_grid(x, y, z, dt, common_min_max)

    filename = f"{index:05d}.png"
    grid.save(grid_folder / filename)
    
    files_list.append({"name": filename, "folder": str(subsubdir)})


def qualitative_analysis_mp(root: Path, new_root: Path, dt: float = 20.0, common_min_max: bool = True):
    new_root.mkdir(parents=True, exist_ok=True)
    grid_folder = new_root / "Grid"
    grid_folder.mkdir(parents=True, exist_ok=True)

    subsubdirs = find_folders(root)

    manager = Manager()
    files_list = manager.list()
    skipped_list = manager.list()

    # Create a partial function with fixed arguments
    worker = partial(process_folder, dt=dt, common_min_max=common_min_max, 
                     grid_folder=grid_folder, files_list=files_list, skipped_list=skipped_list)

    # Use multiprocessing Pool to parallelize folder processing
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(worker, enumerate(subsubdirs)), total=len(subsubdirs), desc="Processing"))

    # Save metadata file
    meta_file = new_root / "metadata.json"
    with open(meta_file, "w") as fp:
        json.dump({
            "dt": dt,
            "files": list(files_list),
            "skipped": list(skipped_list),
        }, fp, indent=2)
    
    print(f"Metadata saved at: {meta_file}")


def qualitative_analysis_threads(root: Path, new_root: Path, dt: float = 20.0, common_min_max: bool = True):
    new_root.mkdir(parents=True, exist_ok=True)
    grid_folder = new_root / "Grid"
    grid_folder.mkdir(parents=True, exist_ok=True)

    subsubdirs = find_folders(root)

    files_list = []
    skipped_list = []

    # Create a partial function with fixed arguments
    worker = partial(process_folder, dt=dt, common_min_max=common_min_max, 
                     grid_folder=grid_folder, files_list=files_list, skipped_list=skipped_list)

    with ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:  # Use more threads since they are I/O bound
        futures = [executor.submit(worker, (i, subsubdir)) for i, subsubdir in enumerate(subsubdirs)]

        for future in tqdm(as_completed(futures), total=len(subsubdirs), desc="Processing"):
            future.result()  # Wait for completion

    meta_file = new_root / "metadata.json"
    with open(meta_file, "w") as fp:
        json.dump({"dt": dt, "files": files_list, "skipped": skipped_list}, fp, indent=2)

    print(f"Metadata saved at: {meta_file}")


if __name__ == "__main__":
    """
    Interpretation of COLORMAP_JET
    RED   ... AOS values lower than GT   ... GT hotter than AOS
    BLUE  ... AOS values higher than GT  ... GT cooler than AOS
    """
    root = Path("/mnt/data/wildfire/IR/Results-batch-1")

    """Threading best suited for disk IO bound tasks."""
    new_root = Path("/mnt/data/wildfire/IR/Analysis")
    qualitative_analysis_threads(root, new_root)  # 00:33

    """Multiprocessing best suited for CPU bound tasks."""
    # new_root = Path("/mnt/data/wildfire/IR/Analysis_mp")
    # qualitative_analysis_mp(root, new_root)  # 10:33
