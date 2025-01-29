import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from ..utils.image import TemperatureDataWithIndices, tone_mapping


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=Path, help="Directory to .TIF files")
    parser.add_argument("tonemapdir", type=Path, help="Directory to save tone maped files.")
    args = parser.parse_args()

    assert args.datadir.exists() and args.datadir.is_dir(), f"{args.datadir} does not exist or is not a directory."
    args.tonemapdir.mkdir(parents=True, exist_ok=True)

    ds = TemperatureDataWithIndices(folder=args.datadir)
    dl = DataLoader(ds, batch_size=16, num_workers=4, drop_last=False)

    for imgs, inds in tqdm(dl, desc="Generating tone mapped images..."):
        for img, idx in zip(imgs, inds):  # HxW
            img_fp: Path = ds.files[int(idx)]
            img: Image.Image = tone_mapping(img[None, :, :], img.min(), img.max())
            new_fp = (args.tonemapdir / img_fp.stem).with_suffix(".png")
            img.save(new_fp)
            