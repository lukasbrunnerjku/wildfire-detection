import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from dataclasses import dataclass, field
import numpy as np
import random
from pathlib import Path
import cv2
from tqdm import tqdm
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
    gated: bool = False
    channels: int = 128
    residual_blocks: int = 2
    residual_channels: int = 32
    embed_dim: int = 64
    # Num. of different env. temp. but
    # if set to 0 => no conditioning used.
    num_cond_embed: int = 31


@dataclass
class DataConf:
    meta_path: Path = Path("C:/IR/data/Batch-1.json")
    val_split: float = 0.1
    key: str = "area_07"
    threshold: float = 0.33


@dataclass
class TrainConf:
    seed: int = 42
    device: str = "cuda"
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
    num_workers: int = 0
    # If use_ssim is False then we penalize the model if it is not
    # able to hallucinate the hidden, never seen, wildfires on the ground.
    # Otherwise, if use_ssim is True we penalize structural similar
    # areas more between GT and AOS, thus emphasising areas that could be corrected.
    use_ssim: bool = False
    model: ModelConf = field(default_factory=ModelConf)
    data: DataConf = field(default_factory=DataConf)
    runs: Path = Path("C:/IR/runs/autoencoder")


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

    def __init__(self, win_size: int = 7, use_ssim: bool = True):
        super().__init__()
        self.mse = nn.MSELoss()
        self.use_ssim = use_ssim
        self.ssim = SSIM(win_size=win_size)
        self._last_ssim = None

    @torch.no_grad()
    def get_ssim(self, aos: Tensor, gt: Tensor) -> Tensor:
        x = TF.normalize(aos, aos.mean(), aos.std())  # Bx1xHxW
        y = TF.normalize(gt, gt.mean(), gt.std())  # Bx1xHxW
        
        gmax = float(torch.max(x.max(), y.max()))
        gmin = float(torch.min(x.min(), y.min()))
        dxy = gmax - gmin

        # More similar regions should be punished more if incorrect.
        ssim = self.ssim(
            x, y,
            data_range=dxy,
            nonnegative_ssim=True,
            size_average=False
        )
        self._last_ssim = ssim  # For visualization purpose.
        return ssim  # Bx1xHxW
    
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
        mse = self.mse(res, tgt)

        if self.use_ssim:  # Try to avoid hallucinations with SSIM weighting.
            ssim = self.get_ssim(aos, gt)
            # TODO: Which is better?
            loss = (ssim * mse).sum() / (ssim.sum() + 1e-5)
            # loss = (ssim * mse).mean()
        else:  # Try to guess hidden structures, force hallucinations.
            loss = mse.mean()
        
        return loss
        

if __name__ == "__main__":
    """
    python -m src.ir.train

    Hyperparameters subject to experiments (mark improvements with + or -):

    [] "skip_constant" in preprocess.py  TODO!!!
    [-] "use_ssim" in criterion
    [++] "num_cond_embed" in model
    [-] "gated" in model

    # TODO delete haecker folder to free disk space, is now on vcnas

    Epoch 61/71, Train Loss: 0.0962, Val Loss: 0.1623
    
    TODO
    [] Rework visualizations: aos, aos+res, gt to see diff. and use same min max temp for tone maps 
    [] Penalty on resulting image not residual, but model predicts residual.
    [] Model directly predict result, not residuals.
    [] GAN loss, perceptual loss etc as in VMambaIR and VQGAN.
    [] More channel width in model?
    [] UNet like skip connections in model?
    """
    conf = OmegaConf.merge(
        OmegaConf.structured(TrainConf()),
        OmegaConf.load("src/configs/ir/ae.yaml"),
    )
    setup_torch(conf.seed)

    conf.runs = conf.runs / str(time.strftime('%Y-%m-%d_%H-%M-%S'))
    print(f"Logging runs to: {conf.runs}")
    conf.runs.mkdir(parents=True, exist_ok=True)

    checkpoints = conf.runs / "checkpoints"
    print(f"Saving checkpoints at: {checkpoints}")
    checkpoints.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(conf, checkpoints / "conf.yaml")

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
    )  # TODO UNet like skip connections?

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

    # NOTE: The validation metric we use is the MSE weighted with
    # structural similarity between GT and AOS temperatures.
    criterion = Criterion(use_ssim=conf.use_ssim).to(accelerator.device)
    val_criterion = Criterion(use_ssim=True).to(accelerator.device)

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
        # Calculate normalization statistics over initial batch.
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
                et = batch["ET"].to(accelerator.device)  # B, >> indices

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
                if model.embedding is not None:
                    # Env. temp. as cond
                    # NOTE: Only working with env. temp. from 0 to Tmax (integers)
                    residual_normalized = model(aos_normalized, et)
                else:
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
                    
                    imgs_list = [
                        (aos + residual).detach().cpu(),
                        gt.cpu(),
                        residual.detach().cpu(),
                        tgt.cpu(),
                    ]
                    colormaps = [
                        cv2.COLORMAP_INFERNO,
                        cv2.COLORMAP_INFERNO,
                        cv2.COLORMAP_JET,
                        cv2.COLORMAP_JET,
                    ]
                    if criterion.use_ssim:
                        imgs_list.append(criterion._last_ssim.cpu())
                        colormaps.append(cv2.COLORMAP_VIRIDIS)
                        
                    grid = create_grid(conf.log_n_images, *imgs_list, colormaps=colormaps)
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
                        et = batch["ET"].to(accelerator.device)  # B, >> indices
                        
                        aos_normalized = (aos - aos_mean) / aos_std
                        tgt_normalized = (tgt - tgt_mean) / tgt_std

                        if model.embedding is not None:
                            residual_normalized = model(aos_normalized, et)
                        else:
                            residual_normalized = model(aos_normalized)
                        val_loss += val_criterion(residual_normalized, tgt_normalized, aos, gt).item()
                        
                avg_val_loss = val_loss / len(val_dataloader)
                writer.add_scalar("val/avg_loss", avg_val_loss, epoch)
                
                if avg_val_loss < best_val_loss:  # Save best model
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), checkpoints / "best.pth")
                
                residual = residual_normalized * tgt_std + tgt_mean
                grid = create_grid(
                    conf.log_n_images,
                    (aos + residual).detach().cpu(),
                    gt.cpu(),
                    residual.detach().cpu(),
                    tgt.cpu(),
                    val_criterion._last_ssim.cpu(),
                    colormaps=[
                        cv2.COLORMAP_INFERNO,
                        cv2.COLORMAP_INFERNO,
                        cv2.COLORMAP_JET,
                        cv2.COLORMAP_JET,
                        cv2.COLORMAP_VIRIDIS,
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
