from PIL import Image
from pathlib import Path
from torchvision.transforms.functional import center_crop

from ..utils.image import pil_make_grid


if __name__ == "__main__":
    dir = Path("/mnt/data/wildfire/thesis")

    imgs = [
        dir / "imgs1_DJI_20231017140103_0109_T.png",
        dir / "imgs1_dec_DJI_20231017140103_0109_T.png",
        dir / "gen_subset_0000118.png",
        dir / "gen_subset_0000008.png",
    ]

    imgs = [center_crop(Image.open(img), [512, 512]) for img in imgs]

    grid = pil_make_grid(imgs, len(imgs))

    grid.save(dir / "grid.png")
