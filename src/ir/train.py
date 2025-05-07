import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
import os
from omegaconf import OmegaConf
from dataclasses import dataclass, field
import numpy as np
import random
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from multiprocessing import cpu_count
import math 
import sys
import time

from .ae import Autoencoder
from .data import AOSDataset
from .similarity import SSIM
from ..utils.image import tone_mapping, pil_make_grid
from ..utils.weight_decay import weight_decay_parameter_split


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


@dataclass
class ModelConf:
    channels: int = 128
    residual_blocks: int = 2
    residual_channels: int = 32
    embed_dim: int = 64
    gated: bool = False


@dataclass
class DataConf:
    meta_path: Path = Path("/mnt/data/wildfire/IR/Batch-1.json")
    val_split: float = 0.1
    key: str = "area_07"
    threshold: float = 0.33


@dataclass
class TrainConf:
    seed: int = 42
    device: str = "cuda"  # TODO: Check if GPUs are free!
    warmup_steps: int = 500
    train_steps: int = 4500
    log_every: int = 50  # [steps]
    vis_every: int = 500  # [steps]
    val_every: int = 5  # [epochs]
    log_n_images: int = 4
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.99
    gradient_clip_val: float = 1.0
    num_workers: int = cpu_count()
    model: ModelConf = field(default_factory=ModelConf)
    data: DataConf = field(default_factory=DataConf)
    runs: Path = Path("/mnt/data/wildfire/IR/runs/autoencoder")
    checkpoints: Path = Path("/mnt/data/wildfire/IR/checkpoints/autoencoder")


def create_grid(ncol: int, *imgs_list: list[Tensor], colormaps=None):
    if colormaps is None:
        colormaps = [cv2.COLORMAP_INFERNO] * len(imgs_list)
    
    tonemapped = []
    for colormap, imgs in zip(colormaps, imgs_list):
        tonemapped.extend(
            [tone_mapping(img, img.min(), img.max(), colormap) for img in imgs[:ncol]]
        )
    grid = pil_make_grid(tonemapped, ncol=ncol)
    return grid


class Criterion(nn.Module):

    def __init__(self, win_size: int):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIM(win_size=win_size)
        self._last_mask = None

    def forward(
        self,
        res: Tensor,
        tgt: Tensor,
        aos: Tensor,
        gt: Tensor,
    ):  # 
        """
        All arguments of shape Bx1xHxW.
        res ... predicted residual
        tgt ... residual target (diff. of aos and gt)
        aos ... input of our model, distorted by AOS
        gt ... if AOS would be error free
        """
        x = TF.normalize(aos, aos.mean(), aos.std())  # Bx1xHxW
        y = TF.normalize(gt, gt.mean(), gt.std())[0]  # Bx1xHxW
        
        gmax = float(torch.max(x.max(), y.max()))
        gmin = float(torch.min(x.min(), y.min()))
        dxy = gmax - gmin

        # More similar regions should be punished more if incorrect.
        ssim = self.ssim(data_range=dxy, nonnegative_ssim=True, size_average=False)
        mse = self.mse(res, tgt)

        loss = (ssim * mse).sum() / (ssim.sum() + 1e-5)  # TODO or .mean() ??
        return loss
        

if __name__ == "__main__":
    """
    python -m src.ir.train
    """
    conf = OmegaConf.merge(
        OmegaConf.structured(TrainConf()),
        OmegaConf.load("src/configs/ir/ae.yaml"),
    )
    setup_torch(conf.seed)

    time_now = str(time.strftime('%Y-%m-%d_%H-%M-%S'))

    conf.runs = conf.runs / time_now
    print(f"Logging runs to: {conf.runs}")
    conf.runs.mkdir(parents=True, exist_ok=True)

    conf.checkpoints = conf.checkpoints / time_now
    print(f"Saving checkpoints at: {conf.checkpoints}")
    conf.checkpoints.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset, val_dataset = AOSDataset(
        None,
        conf.data.meta_path,
        conf.data.key,
        conf.data.threshold,
    ).split(conf.data.val_split)  # Train/Val split
    print(f"Number of training samples: {len(dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=conf.train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=conf.num_workers,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=conf.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # Initialize model, loss function, and optimizer
    if conf.model.gated:
        from .gated import Conv2d, ConvTranspose2d
    else:
        from torch.nn import Conv2d, ConvTranspose2d
        
    kwargs = {k: v for k, v in conf.model.items() if k != "gated"}
    model = Autoencoder(
        in_channel=1,
        conv_cls=Conv2d,
        conv_transpose_cls=ConvTranspose2d,
        **kwargs,
    )

    decay, no_decay = weight_decay_parameter_split(model)
    groups = [
        {"params": decay, "weight_decay": conf.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=conf.learning_rate,
        betas=(conf.beta1, conf.beta2),
        weight_decay=conf.weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, 1e-2, total_iters=conf.warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=conf.train_steps,
                eta_min=conf.min_learning_rate,
            ),
        ],
        milestones=[conf.warmup_steps],
    )

    # Initialize Accelerator and TensorBoard writer
    accelerator = Accelerator(cpu=conf.device == "cpu")
    writer = SummaryWriter(conf.runs)

    criterion = Criterion(win_size=7).to(accelerator.device)

    # Prepare everything with Accelerator
    model, optimizer, dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, dataloader, val_dataloader
    )

    def train_loop():
        epoch = 0
        global_step = 0  # Track training steps
        best_val_loss = float('inf')
        total_steps = conf.train_steps + conf.warmup_steps
        epochs = math.ceil(total_steps / len(dataloader))
        aos_mean, aos_std = None, None
        tgt_mean, tgt_std = None, None
        pbar = tqdm(total=total_steps, desc="Training", unit="step")

        while global_step < conf.train_steps:
            total_loss = 0
            model.train()
            for batch in dataloader:
                # Move to device; Shapes of Bx1xHxW
                aos = batch["AOS"].to(accelerator.device)[:, None, :, :]
                gt = batch["GT"].to(accelerator.device)[:, None, :, :]
                tgt = batch["Residual"].to(accelerator.device)[:, None, :, :]

                if aos_mean is None or aos_std is None:
                    aos_mean = aos.mean()
                    aos_std = aos.std()

                if tgt_mean is None or tgt_std is None:
                    tgt_mean = tgt.mean()
                    tgt_std = tgt.std()
                
                # Normalize the aos input as models learn better on normalized data.
                # Same reason for normalizing the targets, residuals in our case.
                aos_normalized = (aos - aos_mean) / aos_std
                tgt_normalized = (tgt - tgt_mean) / tgt_std

                optimizer.zero_grad()
                residual_normalized = model(aos_normalized)
                loss = criterion(residual_normalized, tgt_normalized, aos, gt)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                
                total_loss += loss.item()

                if global_step % conf.log_every == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    pbar.set_postfix({
                        "loss": f"{loss.item():.3e}",
                        "lr": f"{lr:.3e}",
                    })
                
                if global_step % conf.vis_every == 0:
                    residual = residual_normalized * tgt_std + tgt_mean
                    grid = create_grid(
                        conf.log_n_images,
                        (aos + residual).detach().cpu(),
                        gt.cpu(),
                        residual.detach().cpu(),
                        tgt.cpu(),
                        colormaps=[
                            cv2.COLORMAP_INFERNO,
                            cv2.COLORMAP_INFERNO,
                            cv2.COLORMAP_JET,
                            cv2.COLORMAP_JET,
                        ],
                    )
                    writer.add_image(
                        "train/aos_gt_residual_tgt",
                        np.asarray(grid),
                        global_step,
                        dataformats="HWC",
                    )
                
                pbar.update(1)
                global_step += 1
                if global_step >= conf.train_steps:
                    break
                
            avg_train_loss = total_loss / len(dataloader)
            writer.add_scalar("train/avg_loss", avg_train_loss, epoch)
            
            if epoch % conf.val_every == 0:
                model.eval()
                val_loss = 0
                with torch.inference_mode():
                    for batch in tqdm(val_dataloader, desc="Validation", unit="step", position=0, leave=False):
                        aos = batch["AOS"].to(accelerator.device)[:, None, :, :]
                        gt = batch["GT"].to(accelerator.device)[:, None, :, :]
                        tgt = batch["Residual"].to(accelerator.device)[:, None, :, :]

                        aos_normalized = (aos - aos_mean) / aos_std
                        tgt_normalized = (tgt - tgt_mean) / tgt_std

                        residual_normalized = model(aos_normalized)
                        val_loss += criterion(residual_normalized, tgt_normalized).item()
                        
                avg_val_loss = val_loss / len(val_dataloader)
                writer.add_scalar("val/avg_loss", avg_val_loss, epoch)
                
                if avg_val_loss < best_val_loss:  # Save best model
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), conf.checkpoints / "best.pth")
                
                residual = residual_normalized * tgt_std + tgt_mean
                grid = create_grid(
                    conf.log_n_images,
                    (aos + residual).detach().cpu(),
                    gt.cpu(),
                    residual.detach().cpu(),
                    tgt.cpu(),
                    colormaps=[
                        cv2.COLORMAP_INFERNO,
                        cv2.COLORMAP_INFERNO,
                        cv2.COLORMAP_JET,
                        cv2.COLORMAP_JET,
                    ],
                )
                writer.add_image(
                    "val/aos_gt_residual_tgt",
                    np.asarray(grid),
                    global_step,
                    dataformats="HWC",
                )
                
                sys.stdout.flush()
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            epoch += 1

        writer.close()

    train_loop()
