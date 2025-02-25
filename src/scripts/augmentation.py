import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from typing import Union

from ..utils.image import TemperatureDataWithIndices


def build_semantic_segmentation_target(
    img: Tensor,  # HxW
    fire_thres_temp: float,
    upper_fire_thres_temp: float,
    not_fire_label: int = 0, 
    maybe_fire_label: int = 125,
    is_fire_label: int = 255,
) -> Tensor:
    """
    3-class segmentation task annotation, labels

    not_fire: (-inf, fire_thres_temp]

    maybe_fire: (fire_thres_temp, upper_fire_thres_temp)

    is_fire: [upper_fire_thres_temp, +inf)
    """
    tgt = img.new_full(img.shape, fill_value=not_fire_label, dtype=torch.uint8)
    # import pdb; pdb.set_trace()
    tgt[img > fire_thres_temp] = maybe_fire_label
    tgt[img >= upper_fire_thres_temp] = is_fire_label
    return tgt


def amb_temp_aug(
    img: Tensor,
    amb_temp: float,
    max_sun_temp_inc: float,
    new_amb_temp: float,
    max_amb_temp: float,
):
    delta_amb = new_amb_temp - amb_temp
    new_img = img.clone()
    fire_thres_temp = amb_temp + max_sun_temp_inc
    fire_mask = img >= fire_thres_temp

    # Increas non-fire pixels >0 degree celcius by the temp. delta
    # and make sure the max. ambient temp. condition is not violated.
    amb_mask = torch.logical_and(~fire_mask, new_img > 0)
    new_img[amb_mask] = torch.clip(new_img[amb_mask] + delta_amb, max=max_amb_temp)

    # Set fire-pixels < new ambient temp. to new ambient temp.
    new_img[torch.logical_and(fire_mask, new_img < new_amb_temp)] = new_amb_temp
    return new_img


def build_dataset(
    datadir: Path,
) -> TemperatureDataWithIndices:
    assert datadir.exists() and datadir.is_dir(), f"{datadir} does not exist or is not a directory."
    return TemperatureDataWithIndices(folder=datadir)


def process_dataset(
    ds: Union[TemperatureDataWithIndices, Subset],
    num_workers: int,
    newdatadir: Union[Path, None],
    targetdir: Path,
    amb_temp: float,
    max_sun_temp_inc: float,
    new_amb_temp: float,
    max_amb_temp: float,
    upper_fire_thres_temp: float,
    show_progess: bool,
) -> int:
    num_images_processed = 0

    targetdir.mkdir(parents=True, exist_ok=False)
    if newdatadir is not None:
        newdatadir.mkdir(parents=True, exist_ok=False)

    dl = DataLoader(
        ds, batch_size=1, num_workers=num_workers, drop_last=False
    )

    if isinstance(ds, Subset):
        files = ds.dataset.files
    else:
        files = ds.files

    if show_progess:
        pbar = tqdm(dl, desc="Processing images...")
    else:
        pbar = dl

    for imgs, inds in pbar:
        for img, idx in zip(imgs, inds):  # HxW
            img_fp: Path = files[int(idx)]

            if newdatadir is None:
                fire_thres_temp = amb_temp + max_sun_temp_inc
                tgt_fp = (targetdir / img_fp.stem).with_suffix(".png")
                tgt = build_semantic_segmentation_target(
                    img, fire_thres_temp, upper_fire_thres_temp
                )
                Image.fromarray(tgt.cpu().numpy()).save(tgt_fp)

            else:
                new_img = amb_temp_aug(
                    img,
                    amb_temp,
                    max_sun_temp_inc,
                    new_amb_temp,
                    max_amb_temp,
                )
                
                new_img_fp = newdatadir / img_fp.name
                Image.fromarray(new_img.cpu().numpy()).save(new_img_fp)

                new_fire_thres_temp = new_amb_temp + max_sun_temp_inc
                tgt_fp = (targetdir / img_fp.stem).with_suffix(".png")
                tgt = build_semantic_segmentation_target(
                    new_img, new_fire_thres_temp, upper_fire_thres_temp
                )
                Image.fromarray(tgt.cpu().numpy()).save(tgt_fp)
        
            num_images_processed += 1

    return num_images_processed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=Path, help="Directory to .TIF files")
    parser.add_argument("targetdir", type=Path, help="Directory to save new targets")
    parser.add_argument("--newdatadir", type=Path, default=None, help="Directory to save new .TIF files")
    parser.add_argument("--amb_temp", type=float, default=9, help="Ambient temp. of original data")
    parser.add_argument("--max_sun_temp_inc", type=float, default=15, help="How much hotter can bio mass be under sunlight")
    parser.add_argument("--new_amb_temp", type=float, default=20, help="The new ambient temp. to simulate new data")
    parser.add_argument("--max_amb_temp", type=float, default=30, help="Clip increase by new ambient temp. here")
    parser.add_argument("--upper_fire_thres_temp", type=float, default=60, help="Above this temp. it is fire for sure")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    ds = build_dataset(args.datadir, args.newdatadir, args.targetdir)
    
    process_dataset(
        ds,
        args.num_workers,
        args.newdatadir,
        args.targetdir,
        args.amb_temp,
        args.max_sun_temp_inc,
        args.new_amb_temp,
        args.max_amb_temp,
        args.upper_fire_thres_temp,
        show_progess=True,
    )
