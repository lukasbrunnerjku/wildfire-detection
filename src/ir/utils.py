import cv2
import numpy as np
from pathlib import Path
import torch
from torch import Tensor
from PIL import Image
import re
import random
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch.nn as nn

from ..utils.image import tone_mapping, to_zero_one_range
from .similarity import SSIM, get_ssim
from ..vqvae.perceptual_loss import PerceptualLoss


def setup_torch(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        
def find_folders(root: Path) -> list[Path]:
    subsubdirs = []
    for subdir in sorted(root.glob("*")):
        for subsubdir in sorted(subdir.glob("*")):
            subsubdirs.append(subsubdir)
    return subsubdirs


def read_min_max(file: Path):
    with open(file, "r") as fp:
        line = fp.read()
    min_temp, max_temp = [float(t) for t in line.split(",")]
    return min_temp, max_temp


def write_min_max(file: Path, mi, ma):
    with open(file, "w") as fp:
        fp.write(f"{mi},{ma}")


def load_single_view(image_file: Path, txt_file: Path):
    min_temp, max_temp = read_min_max(txt_file)
    x = torch.from_numpy(cv2.imread(str(image_file), -1).astype(np.float32))  # HxW
    x = ((max_temp - min_temp) * (x / 255)) + min_temp  # in Kelvin
    return x


def numericalSort(full_path: Path):
    file_name = full_path.stem  # no extention
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(file_name)
    parts[1::2] = map(int, parts[1::2])
    return parts


def load_center_view(folder: Path) -> Tensor:
    image_files = sorted((folder / "images").glob("*.png"), key=numericalSort)
    n_files = len(image_files)
    assert n_files % 2 == 1

    txt_files = sorted((folder / "images").glob("*.txt"), key=numericalSort)
    assert len(txt_files) == n_files

    idx = int(n_files / 2)
    x = load_single_view(image_files[idx], txt_files[idx])
    return x


def drone_flight_gif(
    folder: Path,
    min_temp: float, 
    max_temp: float,
    output_path: Path,
    frame_duration: int = 500,
):
    """Take global min./max. temp. from Ground Fire GT image."""
    image_files = sorted((folder / "images").glob("*.png"), key=numericalSort)
    txt_files = sorted((folder / "images").glob("*.txt"), key=numericalSort)

    frames = []
    for idx in range(len(image_files)):
        x = load_single_view(image_files[idx], txt_files[idx])[None, :, :]
        frames.append(tone_mapping(x, min_temp, max_temp, cv2.COLORMAP_INFERNO))

    # Save the first frame and append the rest as a GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,  # Duration of each frame in ms
        loop=0  # Infinite loop
    )
    print(f"GIF created successfully and saved to {output_path}")


def load_xy(dir: Path, normalized: bool = False) -> tuple[Tensor, Tensor]:
    """
    x ... input image; =AOS image; in Kelvin; HxW
    y ... GT image; =floor texture; in Kelvin; HxW
    """
    y = torch.from_numpy(np.array(Image.open(dir / "GT.tiff")))
    min_temp, max_temp = y.min(), y.max()
    if min_temp == max_temp:
        min_max_file = dir / "min_max_temp.txt"
        if min_max_file.exists():
            min_temp, max_temp = read_min_max(min_max_file)
        else:
            min_temps, max_temps = [], []
            files = (dir / "images").glob("*.txt")
            for file in files:
                min_temp, max_temp = read_min_max(file)
                min_temps.append(min_temp)
                max_temps.append(max_temp)

            # Find global min, max from individual views.
            min_temp = min(min_temps)
            max_temp = max(max_temps)
            # Store preprocessed min/max temperatures.
            write_min_max(min_max_file, min_temp, max_temp)

    if normalized:
        x = torch.from_numpy(cv2.imread(
            str(dir / "integrall_normalized_0.png"), -1
        )[:, :, 0].astype(np.float32))  # in [0, 255]; pixel
        min_temp, max_temp = read_min_max(dir / "global_min_max_temp.txt")
    else:
        x = torch.from_numpy(cv2.imread(
            str(dir / "integrall_0.png"), -1
        )[:, :, 0].astype(np.float32))  # in [0, 255]; pixel
    
    x = ((max_temp - min_temp) * (x / 255)) + min_temp  # in Kelvin
        
    return x, y


class Metrics(nn.Module):

    def __init__(self, ):
        super().__init__()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def update(self, preds: Tensor, target: Tensor) -> None:
        new_preds = []
        new_target = []
        for p, t in zip(preds, target):
            min_t, max_t = t.min(), t.max()
            p = to_zero_one_range(p, min=min_t, max=max_t)
            t = to_zero_one_range(t , min=min_t, max=max_t)
            new_preds.append(p)
            new_target.append(t)
        
        new_preds = torch.stack(new_preds, 0)
        new_target = torch.stack(new_target, 0)

        self.psnr.update(new_preds, new_target)
        self.ssim.update(new_preds, new_target)

    def compute(self):
        return {
            "psnr": self.psnr.compute(),
            "ssim": self.ssim.compute(),
        }
    
    def reset(self):
        self.psnr.reset()
        self.ssim.reset()
        

class ImageCriterion(nn.Module):

    def __init__(
        self,
        aos_mean: float,
        aos_std: float,
        # SSIM specifics
        use_ssim: bool = True,
        min_score: float = 0.1,
        # Select loss variations via "objective"
        objective: str = "L2",
        # Perceptual Loss
        resnet_type: str = "resnet18",
        up_to_layer: int = 2,
        perc_weight: float = 0.5,
        l1_weight: float = 1.0,
    ):
        super().__init__()
        self.aos_mean = aos_mean
        self.aos_std = aos_std
        
        self.use_ssim = use_ssim
        self.min_score = min_score
        if self.use_ssim:
            self.ssim = SSIM()
        
        self.objective = objective
        
        self.perc_weight = perc_weight
        self.l1_weight = l1_weight

        if objective == "L2":
            self.mse = nn.MSELoss(reduction="none")
            
        elif objective == "L1":
            self.mae = nn.L1Loss(reduction="none")
        elif objective == "PERC":
            self.mae = nn.L1Loss(reduction="none")
            self.perc = PerceptualLoss(resnet_type, up_to_layer)
        else:
            raise ValueError(f"Unknown {objective=}")
        
        self._last_ssim = None
        
    def maybe_weighted_loss(self, dense_loss: Tensor, aos: Tensor, gt: Tensor) -> Tensor:
        if self.use_ssim:  # Penalize areas with low occlusion more.
            # .min_score ensures all areas get penalized
            ssim = get_ssim(self.ssim, aos, gt).clip(min=self.min_score)
            self._last_ssim = ssim  # For visualization purpose.
            total_loss = (ssim * dense_loss).sum() / (ssim.sum() + 1e-5)
        else:  # Try to guess hidden structures, force hallucinations.
            total_loss = dense_loss.mean()
        return total_loss

    def forward(
        self,
        pred_img_norm: Tensor,
        aos: Tensor,
        gt: Tensor,
    ):
        """
        All arguments of shape Bx1xHxW.
        pred_img_norm ... normalized output image of model
        aos ... input of our model, distorted by AOS
        gt ... if AOS would be error free
        """
        pred_img = self.aos_std * pred_img_norm + self.aos_mean
        
        if self.objective == "L2":
            mse = self.mse(pred_img, gt)
            total_loss = self.maybe_weighted_loss(mse, aos, gt)
            return {"total_loss": total_loss}

        elif self.objective == "L1":
            mae = self.mae(pred_img, gt)
            total_loss = self.maybe_weighted_loss(mae, aos, gt)
            return {"total_loss": total_loss}

        elif self.objective == "PERC":
            l1_loss = self.mae(pred_img, gt).mean()
            gt_norm = (gt - self.aos_mean) / self.aos_std
            perc_loss = self.perc(pred_img_norm, gt_norm)
            total_loss = self.l1_weight * l1_loss + self.perc_weight * perc_loss
            return {
                "total_loss": total_loss,
                "l1_loss": l1_loss,
                "perc_loss": perc_loss,
            }
            
            