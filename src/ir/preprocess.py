from pathlib import Path
import torch
import json
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms.functional as TF
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

"""
Iterate over all files and generate a meta.json file that contains
paths to directories with AOS and GT images
and a list of skipped directories that contain corrupt data.
The non skipped directories have additional information such as
a total ssim score and percentages of ssim area exceeding thresholds
of 0.5, 0.6, 0.7, 0.8, and 0.9. With this meta information we can more
efficiently load the data and train the model later on.
"""

from .utils import find_folders, load_xy


def process_folder(subsubdir, meta_list, skip_constant: bool):
    x, y = load_xy(subsubdir)  # HxW
    # z = y - x  # HxW; Residual

    if skip_constant:
        if torch.allclose(y - y.mean(), 0.0):  # Constant GT wildfire texture?
            return  # Skip

    x = TF.normalize(x[None,...], x.mean(), x.std())[0]  # HxW
    y = TF.normalize(y[None,...], y.mean(), y.std())[0]  # HxW

    gmax = float(max(x.max(), y.max()))
    gmin = float(min(x.min(), y.min()))
    dxy = gmax - gmin

    x, y = x.numpy(), y.numpy()

    _, hm = ssim(x, y, data_range=dxy, full=True, win_size=7)  # HxW; [-inf, 1.0]
    hm = hm.clip(0, 1)
    score = hm.mean()

    n_total = hm.size
    area_05 = (hm > 0.5).sum() / n_total
    area_06 = (hm > 0.6).sum() / n_total
    area_07 = (hm > 0.7).sum() / n_total
    area_08 = (hm > 0.8).sum() / n_total
    area_09 = (hm > 0.9).sum() / n_total

    meta_list.append({
        "folder": str(subsubdir),
        "score": score.item(),
        "area_05": area_05.item(),
        "area_06": area_06.item(),
        "area_07": area_07.item(),
        "area_08": area_08.item(),
        "area_09": area_09.item(),
    })


def process_with_threads(root: Path, skip_constant: bool):
    meta_path = root.parent / f"{root.name}.json" 
    print(f"Meta path: {meta_path}")

    meta_list = []
    subsubdirs = find_folders(root)

    worker = partial(process_folder, meta_list=meta_list, skip_constant=skip_constant)
    with ThreadPoolExecutor(max_workers=2 * cpu_count()) as executor:
        futures = {executor.submit(worker, subsubdir): subsubdir for subsubdir in subsubdirs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    with open(meta_path, "w") as fp:
        json.dump(meta_list, fp, indent=2)
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="C:/IR/data/Batch-1", type=Path)
    parser.add_argument("--skip_constant", action="store_true")  # Constant GT wildfire textures?
    args = parser.parse_args()
    
    process_with_threads(args.root, args.skip_constant)

    meta_path = args.root.parent / f"{args.root.name}.json"
    with open(meta_path, "r") as fp:
        meta_list = json.load(fp)

    subsubdirs = find_folders(args.root)
    n_total = len(subsubdirs)
    n_non_skipped = len(meta_list)
    print(f"Data utilization: {100 * n_non_skipped / n_total}%")

    scores = [item["score"] for item in meta_list]
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, color='blue', alpha=0.7)
    plt.title('Distribution of SSIM Scores')
    plt.xlabel('SSIM Score')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig("ssim_scores_distribution.png")

    def plot_histogram(data, title, xlabel, ylabel, filename):
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=20, color='blue', alpha=0.7)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.savefig(filename)

    plot_histogram(scores, 'Distribution of SSIM Scores', 'SSIM Score', 'Frequency', 'ssim_scores_distribution.png')
    plot_histogram([item["area_05"] for item in meta_list], 'Distribution of SSIM > 0.5 Area Percentage', 'SSIM > 0.5 Area Percentage', 'Frequency', 'ssim_area_05_distribution.png')
    plot_histogram([item["area_06"] for item in meta_list], 'Distribution of SSIM > 0.6 Area Percentage', 'SSIM > 0.6 Area Percentage', 'Frequency', 'ssim_area_06_distribution.png')
    plot_histogram([item["area_07"] for item in meta_list], 'Distribution of SSIM > 0.7 Area Percentage', 'SSIM > 0.7 Area Percentage', 'Frequency', 'ssim_area_07_distribution.png')
    plot_histogram([item["area_08"] for item in meta_list], 'Distribution of SSIM > 0.8 Area Percentage', 'SSIM > 0.8 Area Percentage', 'Frequency', 'ssim_area_08_distribution.png')
    plot_histogram([item["area_09"] for item in meta_list], 'Distribution of SSIM > 0.9 Area Percentage', 'SSIM > 0.9 Area Percentage', 'Frequency', 'ssim_area_09_distribution.png')

    def calculate_percentage_above_threshold(key, meta_list, threshold):
        return 100 * len(list(filter(lambda x: x[key] > threshold, meta_list))) / len(meta_list)
    
    threshold = 0.33
    print(f"Percentage of areas exceeding {threshold=}:")
    for key in ["area_05", "area_06", "area_07", "area_08", "area_09"]:
        percentage = calculate_percentage_above_threshold(key, meta_list, threshold)
        print(f"{key}: {percentage:.2f}%")
    