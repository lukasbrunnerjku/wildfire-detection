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
import json
from typing import Optional

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
    min_learning_rate: float = 1e-6
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
    residual_target: bool = True
    normalize_residual: bool = False
    decay_groups: bool = False
    model: ModelConf = field(default_factory=ModelConf)
    data: DataConf = field(default_factory=DataConf)
    runs: Path = Path("C:/IR/runs/autoencoder")


def create_grid(
    ncol: int,
    *imgs_list: list[Tensor],
    colormaps=None,
    min: Optional[float] = None,
    max: Optional[float] = None,
):
    if colormaps is None:
        colormaps = [cv2.COLORMAP_INFERNO] * len(imgs_list)
    
    tonemapped = []
    for colormap, imgs in zip(colormaps, imgs_list):
        for img in imgs[:ncol]:
            img_min = min if min is not None else img.min()
            img_max = max if max is not None else img.max()
            tonemapped.append(tone_mapping(img, img_min, img_max, colormap))
            
    grid = pil_make_grid(tonemapped, ncol=ncol)
    return grid


class Criterion(nn.Module):

    def __init__(
        self,
        win_size: int = 7,
        use_ssim: bool = True,
        residual_target: bool = True,
        normalize_residual: bool = True,
        aos_mean: Optional[float] = None,
        aos_std: Optional[float] = None,
        res_mean: Optional[float] = None,
        res_std: Optional[float] = None,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.use_ssim = use_ssim
        self.residual_target = residual_target
        self.normalize_residual = normalize_residual
        self.aos_mean = aos_mean
        self.aos_std = aos_std
        self.res_mean = res_mean
        self.res_std = res_std
        self.ssim = SSIM(win_size=win_size)
        self._last_ssim = None

    @torch.no_grad()
    def get_ssim(self, aos: Tensor, gt: Tensor) -> Tensor:
        # More similar regions should be punished more if incorrect.
        x = TF.normalize(aos, aos.mean(), aos.std())  # Bx1xHxW
        y = TF.normalize(gt, gt.mean(), gt.std())  # Bx1xHxW

        dxy = float(torch.max(x.max(), y.max())) - float(torch.min(x.min(), y.min()))

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
        pred_res: Tensor,
        aos: Tensor,
        gt: Tensor,
    ):  # 
        """
        All arguments of shape Bx1xHxW.
        pred_res ... output residual of model
        aos ... input of our model, distorted by AOS
        gt ... if AOS would be error free
        """
        if self.residual_target:
            tgt_res = gt - aos  # := residual target
            if self.normalize_residual:
                tgt_res_norm = (tgt_res - self.res_mean) / self.res_std
                mse = self.mse(pred_res, tgt_res_norm)
            else:
                mse = self.mse(pred_res, tgt_res)
        else:
            if self.normalize_residual:
                pred_img = self.res_std * pred_res + self.res_mean + aos
                mse = self.mse(pred_img, gt)
            else:
                pred_img = pred_res + aos
                mse = self.mse(pred_img, gt)

        if self.use_ssim:  # Try to avoid hallucinations with SSIM weighting.
            ssim = self.get_ssim(aos, gt)
            # loss = (ssim * mse).mean()  TODO: Which is better?
            loss = (ssim * mse).sum() / (ssim.sum() + 1e-5)
        else:  # Try to guess hidden structures, force hallucinations.
            loss = mse.mean()
        
        return loss
        

class MeanStdCalculator:

    def __init__(self):
        self.psum    = torch.tensor([0.0], dtype=torch.float64)
        self.psum_sq = torch.tensor([0.0], dtype=torch.float64)
        self.n_pixel = torch.tensor([0.0], dtype=torch.float64)

    def update(self, x: Tensor):
        B, H, W = x.shape
        self.psum    += x.sum()
        self.psum_sq += (x ** 2).sum()
        self.n_pixel += B * H * W

    def compute(self) -> tuple[Tensor, Tensor]:
        """
        mean := E[X]
        std := sqrt(E[X^2] - (E[X])^2)
        """
        mean = self.psum / self.n_pixel
        var  = (self.psum_sq / self.n_pixel) - (mean ** 2)
        std  = torch.sqrt(var)
        return mean, std
    

def get_mean_std(dl: DataLoader):
    aos_calc = MeanStdCalculator()
    gt_calc = MeanStdCalculator()
    res_calc = MeanStdCalculator()

    for batch in tqdm(dl, "Calculating mean / std..."):
        x = batch["AOS"]  # BxHxW
        y = batch["GT"]  # BxHxW
        z = y - x   

        aos_calc.update(x)
        gt_calc.update(y)
        res_calc.update(z)
        
    x_mean, x_std = aos_calc.compute()    
    y_mean, y_std = gt_calc.compute()    
    z_mean, z_std = res_calc.compute()    

    return {
        "AOS_mean": float(x_mean),
        "AOS_std": float(x_std),
        "GT_mean": float(y_mean),
        "GT_std": float(y_std),
        "RES_mean": float(z_mean),
        "RES_std": float(z_std),
    }


if __name__ == "__main__":
    """
    python -m src.ir.train

    Hyperparameters subject to experiments (mark improvements with + or - or close to equal ~):

    [+] "skip_constant" in preprocess.py  (+ because, on par but without const. harder task)
    [-] "use_ssim" in criterion
    [++] "num_cond_embed" in model
    [-] "gated" in model
    [x] update visuals in tensorboard
    [++] norm stats over total training data
    [~] weight decay not on norms, bias, embeddings
    [~] penalty on image not residual, but model predicts residual
    [~] predict unnormalized residuals

    [] UNet?
    [] More channel width in model?

    TODO
    [] GAN loss, perceptual loss etc as in VMambaIR and VQGAN.
    
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

    fn = conf.data.meta_path.stem  # ie. Batch-1
    th = conf.data.threshold
    th = "_".join(f"{th}".split("."))  # ie. 0.33 => "0_33"
    key = conf.data.key
    stats_path = conf.data.meta_path.parent / f"fn_{fn}_key_{key}_th_{th}.json"
    
    if stats_path.exists():
        print(f"Loading stats from: {stats_path=}")
        with open(stats_path, "r") as fp:
            stats = json.load(fp)
    else:
        print(f"Calculate and save stats to: {stats_path=}")
        stats = get_mean_std(dataloader)
        with open(stats_path, "w") as fp:
            json.dump(stats, fp, indent=4)

    aos_mean = stats["AOS_mean"]
    aos_std = stats["AOS_std"]
    res_mean = stats["RES_mean"]
    res_std = stats["RES_std"]

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

    if conf.decay_groups:
        print("Norms, bias, and embeddings will exclude weight decay.")
        decay, no_decay = weight_decay_parameter_split(model)
        maybe_groups = [
            {"params": decay, "weight_decay": conf.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    else:  # Same settings for all parameters.
        maybe_groups = model.parameters()

    optimizer = optim.AdamW(
        maybe_groups,
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
    criterion = Criterion(
        use_ssim=conf.use_ssim,
        residual_target=conf.residual_target,
        normalize_residual=conf.normalize_residual,
        aos_mean=aos_mean,
        aos_std=aos_std,
        res_mean=res_mean,
        res_std=res_std,
    ).to(accelerator.device)
    val_criterion = Criterion(
        use_ssim=True,
        residual_target=True,
        normalize_residual=True,
        aos_mean=aos_mean,
        aos_std=aos_std,
        res_mean=res_mean,
        res_std=res_std,
    ).to(accelerator.device)  # Keep those for ablation experiments.

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
        pbar = tqdm(total=total_steps, desc="Training", unit="step")

        while global_step <= total_steps:
            total_loss = 0
            model.train()
            for batch in dataloader:
                # Move to device; Shapes of Bx1xHxW
                aos = batch["AOS"].to(accelerator.device)[:, None, :, :]
                gt = batch["GT"].to(accelerator.device)[:, None, :, :]
                et = batch["ET"].to(accelerator.device)  # B, >> indices
                
                aos_normalized = (aos - aos_mean) / aos_std
                
                optimizer.zero_grad()
                if model.embedding is not None:
                    # Env. temp. as cond
                    # NOTE: Only working with env. temp. from 0 to Tmax (integers)
                    pred_res = model(aos_normalized, et)
                else:
                    pred_res = model(aos_normalized)

                loss = criterion(pred_res, aos, gt)
                
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
                    if criterion.normalize_residual:
                        pred = res_std * pred_res + res_mean + aos
                    else:
                        pred = pred_res + aos

                    grid = create_grid(
                        conf.log_n_images,
                        aos.cpu(), pred.detach().cpu(), gt.cpu(),
                        colormaps=[
                            cv2.COLORMAP_INFERNO, cv2.COLORMAP_INFERNO, cv2.COLORMAP_INFERNO
                        ],
                        min=gt.min().cpu(),
                        max=gt.max().cpu(),
                    )
                    writer.add_image(
                        "train/images",
                        np.asarray(grid),
                        global_step,
                        dataformats="HWC",
                    )

                    if criterion.normalize_residual:
                        residual = res_std * pred_res + res_mean
                    else:
                        residual = pred_res

                    tgt = gt - aos
                    grid = create_grid(
                        conf.log_n_images,
                        residual.detach().cpu(), tgt.cpu(),
                        colormaps=[
                            cv2.COLORMAP_JET, cv2.COLORMAP_JET,
                        ],
                        min=tgt.min().cpu(),
                        max=tgt.max().cpu(),
                    )
                    writer.add_image(
                        "train/residuals",
                        np.asarray(grid),
                        global_step,
                        dataformats="HWC",
                    )
                
                pbar.update(1)
                global_step += 1
                if global_step > total_steps:
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
                        et = batch["ET"].to(accelerator.device)  # B, >> indices
                        
                        aos_normalized = (aos - aos_mean) / aos_std

                        if model.embedding is not None:
                            pred_res = model(aos_normalized, et)
                        else:
                            pred_res = model(aos_normalized)

                        if not criterion.normalize_residual:
                            # NOTE: Validation loss expects normalized residuals.
                            pred_res = (pred_res - res_mean) / res_std

                        val_loss += val_criterion(pred_res, aos, gt).item()
                        
                avg_val_loss = val_loss / len(val_dataloader)
                writer.add_scalar("val/avg_loss", avg_val_loss, epoch)
                
                if avg_val_loss < best_val_loss:  # Save best model
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), checkpoints / "best.pth")

                if val_criterion.normalize_residual:
                    pred = res_std * pred_res + res_mean + aos
                else:
                    pred = pred_res + aos

                grid = create_grid(
                    conf.log_n_images,
                    aos.cpu(), pred.detach().cpu(), gt.cpu(),
                    colormaps=[
                        cv2.COLORMAP_INFERNO, cv2.COLORMAP_INFERNO, cv2.COLORMAP_INFERNO
                    ],
                    min=gt.min().cpu(),
                    max=gt.max().cpu(),
                )
                writer.add_image(
                    "val/images",
                    np.asarray(grid),
                    global_step,
                    dataformats="HWC",
                )

                if val_criterion.normalize_residual:
                    residual = res_std * pred_res + res_mean
                else:
                    residual = pred_res
                    
                tgt = gt - aos
                grid = create_grid(
                    conf.log_n_images,
                    residual.detach().cpu(), tgt.cpu(),
                    colormaps=[
                        cv2.COLORMAP_JET, cv2.COLORMAP_JET,
                    ],
                    min=tgt.min().cpu(),
                    max=tgt.max().cpu(),
                )
                writer.add_image(
                    "val/residuals",
                    np.asarray(grid),
                    global_step,
                    dataformats="HWC",
                )

                sys.stdout.flush()
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            epoch += 1

        writer.close()

    train_loop()
