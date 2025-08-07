from torch.utils.data import Dataset, Subset
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
import torch
import numpy as np
import cv2
from typing import Union, Optional
from pathlib import Path
import math


def normalize(x: Tensor) -> Tensor:  # [0, 255] --> [-1., +1.]
    return x.to(torch.float32) * (2.0 / 255.0) - 1.0


def normalize_tif(x: Tensor, mean: float, std: float) -> Tensor:
    return (x - mean) / std


def denormalize(x: Tensor) -> Tensor:  # [-1., +1.] --> [0, 255]
    return ((x + 1.0) * (255.0 / 2.0)).clip(0, 255).to(torch.uint8)


def denormalize_tif(x: Tensor, mean: float, std: float) -> Tensor:
    return x * std + mean


def imwrite(img: torch.Tensor, p: str) -> None:  # CxHxW
    img = denormalize(img.detach()).cpu().numpy().transpose((1, 2, 0)).squeeze()
    img = np.clip(255 * img, 0, 255).astype(np.uint8)
    Image.fromarray(img).convert("P").save(p)


def tif_to_np(file: Union[str, Path]) -> np.ndarray:
    image = np.array(Image.open(file))
    return image


class BaseDataset(Dataset):

    def __init__(self, folder: str, pattern = "*.TIF"):
        self.files = sorted(Path(folder).glob(pattern))

    def __len__(self):
        return len(self.files)
    
    def read_from_file(self, file: Union[str, Path]) -> Tensor:
        image = tif_to_np(file)  # HxW; float32
        return torch.from_numpy(image)


class TemperatureData(BaseDataset):

    def __getitem__(self, idx: int) -> Tensor:
        file = self.files[idx]
        image = self.read_from_file(file)
        return image


class TemperatureDataWithIndices(BaseDataset):

    def __getitem__(self, idx: int) -> Tensor:
        file = self.files[idx]
        image = self.read_from_file(file)
        return image, idx
    

def build_subset_datasets(dataset: BaseDataset, samples_per_subset: int):
    num_total_samples = len(dataset)
    indices = torch.arange(0, num_total_samples)

    datasets = []
    for i in range(0, num_total_samples, samples_per_subset):
        datasets.append(Subset(
            dataset, indices[i:i+samples_per_subset]
        ))
    return datasets


def to_zero_one_range(
    x: Tensor,  # 1xHxW
    min: float,
    max: float,
) -> Tensor:
    x = (x.clamp(min, max) - min) / (max - min)  # [0., 1.]
    return x


def tone_mapping(
    x: Tensor,  # 1xHxW
    min: Optional[float] = None,
    max: Optional[float] = None,
    colormap: int = cv2.COLORMAP_INFERNO,
    return_tensor: bool = False,
) -> Union[Image.Image, Tensor]:
    
    if return_tensor:
        # Useful for evaluation of FID calculation.
        assert x.ndim == 4  # Bx1xHxW
        images = []
        for t in x:  # 1xHxW
            t = to_zero_one_range(t, t.min(), t.max())
            t = (255 * t).to(dtype=torch.uint8).numpy()  # [0, 255]; uint8
            t = cv2.applyColorMap(t[0], colormap)  # HxWxC
            t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
            images.append(t)
        images = np.stack(images)  # BxHxWxC
        images = torch.from_numpy(images).permute(0, 3, 1, 2)  # BxCxHxW
        return images
    else:
        assert x.ndim == 3  # 1xHxW
        x = to_zero_one_range(x, min=min, max=max)
        x = (255 * x).to(dtype=torch.uint8).numpy()  # [0, 255]; uint8
        x = cv2.applyColorMap(x[0], colormap)  # HxWxC
        x = Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))  # HxWxC
        return x
    

def pil_make_grid(
    imgs: list[Image.Image],
    ncol: int = 4,
    padding: int = 2,
    pad_value: int = 0,
) -> Image.Image:
    n_imgs = len(imgs)
    nrow = math.ceil(n_imgs / ncol)
    width = imgs[0].width + padding
    height = imgs[0].height + padding
    grid_width = ncol * width + padding
    grid_height = nrow * height + padding
    grid = Image.new(
        "RGB",
        (grid_width, grid_height),
        (pad_value, pad_value, pad_value),
    )
    img_idx = 0
    for row_idx in range(nrow):
        for col_idx in range(ncol):
            if img_idx >= n_imgs:
                break
            img = imgs[img_idx]
            uper_left_corner = (
                col_idx * width + padding,
                row_idx * height + padding
            )
            grid.paste(img, uper_left_corner)
            img_idx += 1

    return grid


def create_gif_with_text(
    images: list[Image.Image],
    texts: list[str],
    output_path: str,
    frame_duration: int,
    font_size: int = 20,
    text_color: str = "red",
    text_placement: str = "bottom left",
):
    """
    Creates a GIF from a list of PIL Image objects and adds corresponding text to each frame.

    Args:
        images (list of PIL.Image.Image): List of PIL Image objects.
        texts (list of str): List of texts to display on each frame.
        output_path (str): Path where the output GIF will be saved.
        frame_duration (int): Duration of each frame in milliseconds.
    """
    try:
        if len(images) != len(texts):
            raise ValueError("The number of images and texts must be the same.")

        # Create frames with text
        frames = []
        for image, text in zip(images, texts):
            frame = image.copy()
            draw = ImageDraw.Draw(frame)

            # Add text to the frame (bottom-left corner)
            font = ImageFont.load_default(font_size)

            left, top, right, bottom = font.getbbox(text)
            text_height = bottom - top
            text_width = right - left

            if text_placement == "bottom left":
                text_x = 10
                text_y = frame.height - text_height - 10
            elif text_placement == "top right":
                text_x = frame.width - text_width
                text_y = text_height - 10
            else:
                raise ValueError(f"Unknown value: {text_placement=}")
            
            draw.text((text_x, text_y), text, font=font, fill=text_color)
            
            frames.append(frame)

        # Save the first frame and append the rest as a GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,  # Duration of each frame in ms
            loop=0  # Infinite loop
        )
        print(f"GIF created successfully and saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


def create_grid(
    ncol: int,
    *imgs_list: list[Tensor],
    colormaps=None,
    min_max: Optional[tuple[float, float]] = None,
    min_max_idx: Optional[int] = None,
    percentiles: Optional[tuple[float, float]] = None,
):
    """
    Either all min/max related arguments are None, or...
    min_max is not None and min_max_idx is None or...
    min_max is None and min_max_idx is not None...

    If min_max is given directly the "percentiles" argument has no effect.

    Even with percentiles=(0.0, 99.9) we lose the contrast to the peak of
    ground fire in the tone mapped image. Thus, recommandation against the
    use of percentiles.
    """
    if colormaps is None:
        colormaps = [cv2.COLORMAP_INFERNO] * len(imgs_list)
    
    def get_min_max(x: Tensor, percentiles: Optional[tuple[float, float]] = None):
        if percentiles is not None:
            flat = x.numpy().reshape(-1)
            _min = float(np.percentile(flat, percentiles[0]))
            _max = float(np.percentile(flat, percentiles[1]))
        else:
            _min, _max = float(x.min()), float(x.max())
        
        return _min, _max

    if min_max_idx is not None:
        imgs = imgs_list[min_max_idx][:ncol]
        min_max = get_min_max(imgs, percentiles)
    
    tonemapped = []
    for colormap, imgs in zip(colormaps, imgs_list):
        for img in imgs[:ncol]:
            if min_max is None:
                img_min_max = get_min_max(img, percentiles)
            else:
                img_min_max = min_max

            tonemapped.append(tone_mapping(img, *img_min_max, colormap))
            
    grid = pil_make_grid(tonemapped, ncol=ncol)
    return grid


def to_image(pred_res_or_img, aos, aos_std, aos_mean, conf, res_std, res_mean):
    """Model output and configuration. Build prediction image."""
    if conf.predict_image:
        pred = aos_std * pred_res_or_img + aos_mean
    else:
        if conf.normalize_residual:
            pred = res_std * pred_res_or_img + res_mean + aos
        else:
            pred = pred_res_or_img + aos
    
    return pred


def log_images(
    writer,
    global_step,
    phase: str,
    pred_res_or_img,
    aos,
    aos_std,
    aos_mean,
    res_std,
    res_mean,
    gt,
    conf,
):
    pred = to_image(pred_res_or_img, aos, aos_std, aos_mean, conf, res_std, res_mean)

    grid = create_grid(
        conf.log_n_images,
        aos.cpu(), pred.detach().cpu(), gt.cpu(),
        colormaps=[
            cv2.COLORMAP_INFERNO, cv2.COLORMAP_INFERNO, cv2.COLORMAP_INFERNO
        ],
        min_max_idx=2,
    )
    writer.add_image(
        f"{phase}/images",
        np.asarray(grid),
        global_step,
        dataformats="HWC",
    )
    
