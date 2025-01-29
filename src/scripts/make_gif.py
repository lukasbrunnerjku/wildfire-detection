from pathlib import Path
from PIL import Image
import torch
import cv2

from ..utils.image import create_gif_with_text, tif_to_np, tone_mapping, pil_make_grid
from .augmentation import build_semantic_segmentation_target, amb_temp_aug


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", type=Path, help="Directory to save gif file")
    parser.add_argument("tif_fp", type=Path, help="Path to .TIF file")
    parser.add_argument("start_amb_temp", type=int)
    parser.add_argument("stop_amb_temp", type=int)
    parser.add_argument("--step_amb_temp", type=int, default=1)
    parser.add_argument("--amb_temp", type=float, default=9, help="Ambient temp. of original data")
    parser.add_argument("--max_sun_temp_inc", type=float, default=15, help="How much hotter can bio mass be under sunlight")
    parser.add_argument("--max_amb_temp", type=float, default=30, help="Clip increase by new ambient temp. here")
    parser.add_argument("--upper_fire_thres_temp", type=float, default=60, help="Above this temp. it is fire for sure")
    args = parser.parse_args()

    gif_fp: Path = (args.save_dir / args.tif_fp.stem).with_suffix(".gif")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    img = torch.from_numpy(tif_to_np(args.tif_fp))  # HxW; float32

    fire_thres_temp = args.amb_temp + args.max_sun_temp_inc
    tgt = build_semantic_segmentation_target(img, fire_thres_temp, args.upper_fire_thres_temp)
    img2: Image.Image = tone_mapping(img[None, :, :], img.min(), img.max())
    tgt2: Image.Image = tone_mapping(
        tgt[None, :, :].to(dtype=torch.float32),
        float(tgt.min()),
        float(tgt.max()),
        colormap=cv2.COLORMAP_BONE,
    )
    
    images, texts = [], []
    for new_amb_temp in range(args.start_amb_temp, args.stop_amb_temp, args.step_amb_temp):
        new_img = amb_temp_aug(
            img, args.amb_temp, args.max_sun_temp_inc, new_amb_temp, args.max_amb_temp
        )
        new_img2: Image.Image = tone_mapping(
            new_img[None, :, :], new_img.min(), new_img.max()
        )
        new_fire_thres_temp = new_amb_temp + args.max_sun_temp_inc
        new_tgt = build_semantic_segmentation_target(img, new_fire_thres_temp, args.upper_fire_thres_temp)
        new_tgt2: Image.Image = tone_mapping(
            new_tgt[None, :, :].to(dtype=torch.float32),
            float(new_tgt.min()),
            float(new_tgt.max()),
            colormap=cv2.COLORMAP_BONE,
        )
        image = pil_make_grid([img2, tgt2, new_img2, new_tgt2], ncol=2, padding=0)
        text = f"Amb.: {args.amb_temp:02d} >> {new_amb_temp:02d} Thres.: {fire_thres_temp:02d} >> {new_fire_thres_temp:02d}"
        images.append(image)
        texts.append(text)

    create_gif_with_text(
        images,
        texts,
        str(gif_fp),
        500,
        text_placement="top right",
    )
