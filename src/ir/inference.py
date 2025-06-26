from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import cv2
import torch
import json

from .train import (
    AOSDataset,
    create_grid,
    setup_torch,
    drone_flight_gif,
    load_center_view,
)
from .mambaout import MambaOutIR


def load_from(checkpoint: Path) -> MambaOutIR:
    model = MambaOutIR(
        1,
        (64, 128, 256),
        (2, 2, 3),
        num_class_embeds=31,
        oss_refine_blocks=2,
        local_embeds=False,
        drop_path=0.2,
        with_stem=True,
    )
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=Path("/home/lbrunn/projects/irdata/inference"))
    parser.add_argument("--meta_path", default=Path("/home/lbrunn/projects/irdata/data/Batch-1.json"))
    parser.add_argument("--key", default="area_07")
    parser.add_argument("--threshold", type=float, default=0.33)
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--log_n_images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    conf = parser.parse_args()

    setup_torch(conf.seed)
    
    checkpoint = Path("/home/lbrunn/projects/irdata/runs/2025-06-04_11-11-16/checkpoints/best.pth")
    model = load_from(checkpoint)
    exit()
    # Load dataset
    dataset, val_dataset = AOSDataset(
        None,
        conf.meta_path,
        conf.key,
        conf.threshold,
        conf.normalized,
    ).split(conf.val_split, conf.seed)  # Train/Val split
    print(f"Number of training samples: {len(dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    
    # TODO: Include stats into checkpoint, and all other model relevant params!
    fn = conf.data.meta_path.stem  # ie. Batch-1
    th = conf.data.threshold
    th = "_".join(f"{th}".split("."))  # ie. 0.33 => "0_33"
    key = conf.data.key
    nm = int(conf.data.normalized)  # 0 ... not normalized
    stats_path = conf.data.meta_path.parent / f"fn_{fn}_key_{key}_th_{th}_nm_{nm}.json"
    print(f"Loading stats from: {stats_path=}")
    with open(stats_path, "r") as fp:
        stats = json.load(fp)

    aos_mean = stats["AOS_mean"]
    aos_std = stats["AOS_std"]

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
    