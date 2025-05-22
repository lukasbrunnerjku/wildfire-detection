from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import cv2

from .train import (
    AOSDataset,
    create_grid,
    setup_torch,
    drone_flight_gif,
    load_center_view,
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=Path("/mnt/data/wildfire/IR/Inference"))
    parser.add_argument("--meta_path", default=Path("/mnt/data/wildfire/IR/Batch-1.json"))
    parser.add_argument("--key", default="area_07")
    parser.add_argument("--threshold", type=float, default=0.33)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--log_n_images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    conf = parser.parse_args()

    setup_torch(conf.seed)

    # Load dataset
    dataset, val_dataset = AOSDataset(
        None,
        conf.meta_path,
        conf.key,
        conf.threshold,
    ).split(conf.val_split, conf.seed)  # Train/Val split
    print(f"Number of training samples: {len(dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=conf.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    for batch in tqdm(val_dataloader, desc="Validation", unit="step", position=0, leave=False):
        pass

    aos = batch["AOS"][:, None, :, :]
    gt = batch["GT"][:, None, :, :]
    et = batch["ET"]  # B, >> indices
    idx = batch["IDX"]  # B,
    
    folders = []
    for i in idx[:conf.log_n_images]:
        folders.append(val_dataset.folders[i])

    for i, folder in enumerate(folders):
        output_path = conf.output_path / f"{i:02d}.gif"
        min_temp = float(gt[i].min().cpu())
        max_temp = float(gt[i].max().cpu())
        drone_flight_gif(folder, min_temp, max_temp, output_path)

    cev = [load_center_view(f)[None, :, :] for f in folders]

    grid = create_grid(
        conf.log_n_images,
        aos.cpu(), cev, gt.cpu(),
        colormaps=[
            cv2.COLORMAP_INFERNO, cv2.COLORMAP_INFERNO, cv2.COLORMAP_INFERNO
        ],
        min_max_idx=2,
    )
    grid.save(conf.output_path / "aos_cev_gt.png")
    