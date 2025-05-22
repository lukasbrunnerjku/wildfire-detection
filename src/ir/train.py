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
import math 
import sys
import time
import json
from typing import Optional
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from .ae import Autoencoder
from .unet import UNet
from .mamba import VmambaIR
from .data import AOSDataset
from .similarity import SSIM
from .utils import load_center_view, drone_flight_gif
from ..utils.image import tone_mapping, pil_make_grid, to_zero_one_range
from ..utils.weight_decay import weight_decay_parameter_split
from ..vqvae.perceptual_loss import PerceptualLoss
from ..vqvae.gan import AdversarialLoss

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
    num_cond_embed: int = 0 # 31
    arch: str = "" # "mamba", "unet", ""


@dataclass
class DataConf:
    meta_path: Path = Path("/mnt/data/wildfire/IR/Batch-1.json") # Path("C:/IR/data/Batch-1.json")
    val_split: float = 0.1
    key: str = "area_07"
    threshold: float = 0.33


@dataclass
class TrainConf:
    seed: int = 42
    device: str = "cuda"
    warmup_steps: int = 500
    train_steps: int = 9500
    log_every: int = 50  # [steps]
    vis_every: int = 500  # [steps]
    val_every: int = 5  # [epochs]
    log_n_images: int = 4
    train_batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 3e-4 # 1e-3
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4  # 1e-2
    beta1: float = 0.9
    beta2: float = 0.999 # 0.99
    gradient_clip_val: float = 1.0
    num_workers: int = 0
    # If use_ssim is False then we penalize the model if it is not
    # able to hallucinate the hidden, never seen, wildfires on the ground.
    # Otherwise, if use_ssim is True we penalize structural similar
    # areas more between GT and AOS, thus emphasising areas that could be corrected.
    use_ssim: bool = False
    residual_target: bool = True  # NOTE: has no effect if predict_imag=True
    normalize_residual: bool = False  # NOTE: has no effect if predict_imag=True
    decay_groups: bool = False
    predict_image: bool = True
    img_sz: tuple[int] = (512, 512)
    # Select loss variations via "objective"
    objective: str = "L2"  # L2, L1, PERC, GAN; default=L2
    perc_weight: float = 0.5
    l1_weight: float = 1.0
    # Perceptual Loss (predict_image must be True!)
    resnet_type: str = "resnet18"
    up_to_layer: Optional[int] = 2
    # GAN Loss (predict_image must be True!)
    start_epoch: int = 10
    disc_weight: float = 0.1
    adaptive_weight: bool = False
    loss_type: str = "hinge"
    lecam_reg_weight: float = 1e-3
    ema_decay: float = 0.99
    n_layers: int = 2
    ### ###
    model: ModelConf = field(default_factory=ModelConf)
    data: DataConf = field(default_factory=DataConf)
    runs: Path = Path("/mnt/data/wildfire/IR/runs/new")  # C:/IR/runs/autoencoder


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


@torch.no_grad()
def get_ssim(model, aos: Tensor, gt: Tensor) -> Tensor:
    """More similar regions should be punished more if incorrect."""
    x = TF.normalize(aos, aos.mean(), aos.std())  # Bx1xHxW
    y = TF.normalize(gt, gt.mean(), gt.std())  # Bx1xHxW

    dxy = float(torch.max(x.max(), y.max())) - float(torch.min(x.min(), y.min()))

    ssim = model(
        x, y,
        data_range=dxy,
        nonnegative_ssim=True,
        size_average=False
    )
    return ssim  # Bx1xHxW


class ImageCriterion(nn.Module):

    def __init__(
        self,
        win_size: int = 7,
        use_ssim: bool = True,
        aos_mean: Optional[float] = None,
        aos_std: Optional[float] = None,
        # Select loss variations via "objective"
        objective: str = "L1",
        # Perceptual Loss
        resnet_type: str = "resnet18",
        up_to_layer: Optional[int] = 2,
        # GAN loss
        start_epoch: int = 10,
        disc_weight: float = 0.1,
        adaptive_weight: bool = False,
        loss_type: str = "hinge",
        lecam_reg_weight: float = 1e-3,
        ema_decay: float = 0.99,
        n_layers: int = 2,
    ):
        super().__init__()
        self.use_ssim = use_ssim
        self.aos_mean = aos_mean
        self.aos_std = aos_std
        self.objective = objective

        if objective == "L2":
            self.mse = nn.MSELoss(reduction="none")
            self.ssim = SSIM(win_size=win_size)
        elif objective == "L1":
            self.mae = nn.L1Loss(reduction="none")
        elif objective == "PERC":
            self.mae = nn.L1Loss(reduction="none")
            self.perc = PerceptualLoss(resnet_type, up_to_layer)
        elif objective == "GAN":
            self.mae = nn.L1Loss(reduction="none")
            self.perc = PerceptualLoss(resnet_type, up_to_layer)
            assert start_epoch >= 0
            self.adv = AdversarialLoss(
                in_channels=1,
                disc_weight=disc_weight,
                start_step=start_epoch,
                adaptive_weight=adaptive_weight,
                loss_type=loss_type,
                lecam_reg_weight=lecam_reg_weight,
                ema_decay=ema_decay,
                n_layers=n_layers,
            )
        else:
            raise ValueError(f"Unknown {objective=}")
        
        self._last_ssim = None

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
        
            if self.use_ssim:  # Try to avoid hallucinations with SSIM weighting.
                ssim = get_ssim(self.ssim, aos, gt)
                self._last_ssim = ssim  # For visualization purpose.
                total_loss = (ssim * mse).sum() / (ssim.sum() + 1e-5)
            else:  # Try to guess hidden structures, force hallucinations.
                total_loss = mse.mean()
            
            return {"total_loss": total_loss}

        elif self.objective == "L1":
            total_loss = self.mae(pred_img, gt).mean()
            return {"total_loss": total_loss}

        elif self.objective == "PERC":

            return {
                "total_loss": total_loss,
            }
        
        elif self.objective == "GAN":

            return {
                "total_loss": total_loss,
            }
    

class ResidualCriterion(nn.Module):

    def __init__(
        self,
        win_size: int = 7,
        use_ssim: bool = True,
        residual_target: bool = True,
        normalize_residual: bool = True,
        res_mean: Optional[float] = None,
        res_std: Optional[float] = None,
    ):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.use_ssim = use_ssim
        self.residual_target = residual_target
        self.normalize_residual = normalize_residual
        self.res_mean = res_mean
        self.res_std = res_std
        self.ssim = SSIM(win_size=win_size)
        self._last_ssim = None
    
    def forward(
        self,
        pred_res: Tensor,
        aos: Tensor,
        gt: Tensor,
    ):
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
            ssim = get_ssim(self.ssim, aos, gt)
            self._last_ssim = ssim  # For visualization purpose.
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


def to_image(pred_res_or_img, aos, aos_std, aos_mean, conf):
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
    gt,
    conf,
):
    pred = to_image(pred_res_or_img, aos, aos_std, aos_mean, conf)

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

    if conf.predict_image:
        residual = gt - pred
    else:
        if conf.normalize_residual:
            residual = res_std * pred_res_or_img + res_mean
        else:
            residual = pred_res_or_img

    tgt = gt - aos
    grid = create_grid(
        conf.log_n_images,
        residual.detach().cpu(), tgt.cpu(),
        colormaps=[
            cv2.COLORMAP_JET, cv2.COLORMAP_JET,
        ],
        min_max_idx=1,
    )
    writer.add_image(
        f"{phase}/residuals",
        np.asarray(grid),
        global_step,
        dataformats="HWC",
    )


if __name__ == "__main__":
    """
    python -m src.ir.train

    Hyperparameters subject to experiments (mark improvements with + or - or close to equal ~):

    [+] "skip_constant" in preprocess.py  (+ because, on par but without const. harder task)
    [++] "num_cond_embed" in model
    [-] "gated" in model
    [x] update visuals in tensorboard
    [++] norm stats over total training data
    [~] weight decay not on norms, bias, embeddings
    [~] penalty on image not residual, but model predicts residual
    [~] predict unnormalized residuals
    [~+] train batch size 8 instead of 16 & AMP & double steps (to have seen same amount of data)
    [--] UNet stride 4 with condition
    [--] "use_ssim" in criterion
    [x] SSIM and PSNR on tone mapped images to be compatible with other IR methods
    [-] unet with residual prediction
    [-] is unet better suited if predicting image directly instead of residual?
    => no matter what unet is worse till now

    [---????] let models predict image directly instead of residual prediction TODO: buggy visu residual?

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
    ).split(conf.data.val_split, conf.seed)  # Train/Val split
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
    if conf.model.arch == "mamba":
        num_class_embeds = None if conf.model.num_cond_embed <= 0 else conf.model.num_cond_embed
        model = VmambaIR(
            1,
            conf.img_sz,
            (16, 32, 64),
            (2, 2, 2),
            num_class_embeds,
            1,
            conf.predict_image,
        )
    elif conf.model.arch == "unet":
        block_out_channels = [32, 32, 64, 64, 128]  # Stride 16

        if conf.model.num_cond_embed > 0:
            # A NOTE on "block_out_channels", a
            # channel below 32 would not work due to group norm of KDowblock2D in "get_down_block"
            # receives not the number of groups as parameter instead uses always default of 32
            # ie. 16 instead of 32 for first channel ==> ZeroDivisionError: integer division or modulo by zero 
            # TODO: Even more customization required to make a more efficient yet powerfull UNet arch.
            model = UNet(
                in_channels=1,
                out_channels=1,
                down_block_types=["KDownBlock2D"] * len(block_out_channels),
                up_block_types=["KUpBlock2D"] * len(block_out_channels),
                block_out_channels=block_out_channels,  
                num_class_embeds=conf.model.num_cond_embed,
                resnet_time_scale_shift="ada_group",
                norm_num_groups=16,
            )
        else:
            model = UNet(
                in_channels=1,
                out_channels=1,
                down_block_types=["DownBlock2D"] * len(block_out_channels),
                up_block_types=["UpBlock2D"] * len(block_out_channels),
                block_out_channels=block_out_channels,
                norm_num_groups=16,
            )
    else:
        if conf.model.gated:
            from .gated import Conv2d, ConvTranspose2d
        else:
            from torch.nn import Conv2d, ConvTranspose2d
            
        kwargs = {k: v for k, v in conf.model.items() if k not in ("gated", "arch")}
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
    accelerator = Accelerator(cpu=conf.device == "cpu", mixed_precision="fp16")
    writer = SummaryWriter(conf.runs)

    # NOTE: The validation metric we use is the MSE weighted with
    # structural similarity between GT and AOS temperatures.
    if conf.predict_image:
        criterion = ImageCriterion(
            use_ssim=conf.use_ssim,
            aos_mean=aos_mean,
            aos_std=aos_std,
            objective=conf.objective,
            resnet_type=conf.resnet_type,
            up_to_layer=conf.up_to_layer,
            start_epoch=conf.start_epoch,
            disc_weight=conf.disc_weight,
            adaptive_weight=conf.adaptive_weight,
            loss_type=conf.loss_type,
            lecam_reg_weight=conf.lecam_reg_weight,
            ema_decay=conf.ema_decay,
            n_layers=conf.n_layers,
        )
        val_criterion = ImageCriterion(
            use_ssim=True,
            aos_mean=aos_mean,
            aos_std=aos_std,
        ).to(accelerator.device)
    else:
        criterion = ResidualCriterion(
            use_ssim=conf.use_ssim,
            residual_target=conf.residual_target,
            normalize_residual=conf.normalize_residual,
            res_mean=res_mean,
            res_std=res_std,
        )
        val_criterion = ResidualCriterion(
            use_ssim=True,
            residual_target=conf.residual_target,
            normalize_residual=conf.normalize_residual,
            res_mean=res_mean,
            res_std=res_std,
        ).to(accelerator.device)  

    # Metrics for ablation experiments.
    metrics = Metrics().to(accelerator.device)

    # Prepare everything with Accelerator
    model, criterion, optimizer, dataloader, val_dataloader = accelerator.prepare(
        model, criterion, optimizer, dataloader, val_dataloader
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
                    pred_res_or_img = model(aos_normalized, et)
                else:
                    pred_res_or_img = model(aos_normalized)

                loss = criterion(pred_res_or_img, aos, gt)
                if not isinstance(loss, torch.Tensor):
                    loss = loss["total_loss"] 
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                
                total_loss += loss.item()

                pred = to_image(pred_res_or_img.detach(), aos, aos_std, aos_mean, conf)
                metrics.update(pred, gt)

                if global_step % conf.log_every == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    pbar.set_postfix({
                        "loss": f"{loss.item():.3e}",
                        "lr": f"{lr:.3e}",
                    })
                
                if global_step % conf.vis_every == 0:
                    log_images(
                        writer,
                        global_step,
                        "train",
                        pred_res_or_img,
                        aos,
                        aos_std,
                        aos_mean,
                        gt,
                        conf,
                    )
                
                pbar.update(1)
                global_step += 1
                if global_step > total_steps:
                    break
                
            avg_train_loss = total_loss / len(dataloader)
            writer.add_scalar("train/avg_loss", avg_train_loss, epoch)

            r = metrics.compute()
            writer.add_scalar("train/psnr", r["psnr"].item(), epoch)
            writer.add_scalar("train/ssim", r["ssim"].item(), epoch)
            metrics.reset()

            if epoch % conf.val_every == 0:
                model.eval()
                val_loss = 0
                with torch.inference_mode():
                    for batch in tqdm(val_dataloader, desc="Validation", unit="step", position=0, leave=False):
                        aos = batch["AOS"].to(accelerator.device)[:, None, :, :]
                        gt = batch["GT"].to(accelerator.device)[:, None, :, :]
                        et = batch["ET"].to(accelerator.device)  # B, >> indices
                        idx = batch["IDX"]  # B,

                        aos_normalized = (aos - aos_mean) / aos_std

                        if model.embedding is not None:
                            pred_res_or_img = model(aos_normalized, et)
                        else:
                            pred_res_or_img = model(aos_normalized)

                        loss = val_criterion(pred_res_or_img, aos, gt)
                        if not isinstance(loss, torch.Tensor):
                            total_loss = loss["total_loss"]
                        val_loss += total_loss.item()

                        pred = to_image(pred_res_or_img, aos, aos_std, aos_mean, conf)
                        metrics.update(pred, gt)
                        
                avg_val_loss = val_loss / len(val_dataloader)
                writer.add_scalar("val/avg_loss", avg_val_loss, epoch)

                r = metrics.compute()
                writer.add_scalar("val/psnr", r["psnr"].item(), epoch)
                writer.add_scalar("val/ssim", r["ssim"].item(), epoch)
                metrics.reset()
                
                if avg_val_loss < best_val_loss:  # Save best model
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), checkpoints / "best.pth")

                if epoch == 0:
                    folders = []
                    for i in idx[:conf.log_n_images]:
                        folders.append(val_dataset.folders[i])
                    
                    for i, folder in enumerate(folders):
                        output_path = conf.runs / f"{i:02d}.gif"
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
                    writer.add_image(
                        "center_view",
                        np.asarray(grid),
                        0,
                        dataformats="HWC",
                    )

                log_images(
                    writer,
                    global_step,
                    "val",
                    pred_res_or_img,
                    aos,
                    aos_std,
                    aos_mean,
                    gt,
                    conf,
                )

                sys.stdout.flush()
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            epoch += 1

        writer.close()

    train_loop()
