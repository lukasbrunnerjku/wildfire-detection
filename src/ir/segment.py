import pandas as pd
from pathlib import Path
import numpy as np

from .utils import load_xy


def sample_evenly(df: pd.DataFrame, n: int = 10, key: str = "tree_density"):
    df_sorted = df.sort_values(by=key)

    # Get min and max of tree_density
    min_density = df_sorted[key].min()
    max_density = df_sorted[key].max()

    # Generate 10 target values spaced across the range
    target_values = np.linspace(min_density, max_density, n, dtype=int)

    # For each target, find the closest row (within a tolerance)
    sampled_rows = []
    for val in target_values:
        closest_row = df_sorted.iloc[(df_sorted[key] - val).abs().argsort()[:1]]
        sampled_rows.append(closest_row)

    # Combine into a final DataFrame (and remove potential duplicates)
    sampled_df = pd.concat(sampled_rows).drop_duplicates()

    return sampled_df


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    
    from .similarity import SSIM, get_ssim, visualize
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/data/wildfire/IR/root")
    args = parser.parse_args()
    
    root = Path(args.root)
    csv_files = list(root.glob("Batch-*.csv"))
    
    outdir = root.parent / "ssim"
    outdir.mkdir(exist_ok=True)
    
    model = SSIM()
    
    for csv_file in tqdm(csv_files):
        df = pd.read_csv(csv_file)

        df = sample_evenly(df)     
        
        for row in df.itertuples(index=False):
            folder = root / row.aos_folder
            density = row.tree_density
            
            aos, gt = load_xy(folder, normalized=True)
            hm = get_ssim(model, aos[None, None, ...], gt[None, None, ...])[0, 0, ...]  # HxW
            visualize(str(outdir / f"{csv_file.stem}_{density:02d}.png"), aos, gt, hm)
            