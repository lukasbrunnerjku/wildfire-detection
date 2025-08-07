import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from omegaconf import OmegaConf
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import cv2
import os
from tqdm import tqdm
import math 
import sys
import time
import json
from typing import Optional

from .mambaout import MambaOutIR
from .data import get_mean_std, LargeAOSDatsetBuilder
from .utils import (
    load_center_view,
    drone_flight_gif,
    setup_torch,
    Metrics,
    ImageCriterion,
)
from ..utils.image import create_grid, log_images, to_image
from ..utils.weight_decay import weight_decay_parameter_split


@dataclass
class ModelConf:
    # Num. of different env. temp. but
    # if set to 0 => no conditioning used.
    num_cond_embed: int = 31 # 0 or 31
    arch: str = "mambaout" # "mambaout", "mamba", "unet"


@dataclass
class DataConf:
    root: Path = Path("/mnt/data/wildfire/IR/root")
    normalized: bool = True
    drop_uniform: bool = True
    img_sz: Optional[int] = 512 # 128, 512


@dataclass
class TrainConf:
    seed: int = 42
    device: str = "cuda"
    warmup_steps: int = 500
    train_steps: int = 50000
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
    num_workers: int = 16
    use_ssim: bool = False  # TODO: Test its effectiveness!
    residual_target: bool = True  # NOTE: has no effect if predict_imag=True
    normalize_residual: bool = False  # NOTE: has no effect if predict_imag=True
    decay_groups: bool = True
    predict_image: bool = True
    add_residual: bool = True
    # Select loss variations via "objective"
    objective: str = "PERC"  # L2, L1, PERC, GAN; default=L2
    perc_weight: float = 2.0
    l1_weight: float = 1.0
    # Perceptual Loss (predict_image must be True!)
    resnet_type: str = "resnet18"
    up_to_layer: Optional[int] = 2
    model: ModelConf = field(default_factory=ModelConf)
    data: DataConf = field(default_factory=DataConf)
    runs: Path = Path("/mnt/data/wildfire/IR/runs/large")  
    

if __name__ == "__main__":
    """
    python -m src.ir.train_large
    
    tensorboard --logdir=/mnt/data/wildfire/IR/runs/large
    """
    print(f"Running from working directory: {os.getcwd()}")
    
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
    
    builder = LargeAOSDatsetBuilder(conf.data.root)
    train_ds = builder.get_dataset(
        "train", normalized=conf.data.normalized, drop_uniform=conf.data.drop_uniform
    )
    test_ds = builder.get_dataset(
        "test", normalized=conf.data.normalized, drop_uniform=conf.data.drop_uniform
    )
    
    mean_std_file = conf.data.root / f"mean_std_n{int(conf.data.normalized)}_du{int(conf.data.drop_uniform)}.json"
    if not mean_std_file.exists():
        train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=16)
        train_stats = get_mean_std(train_dl)
        with open(mean_std_file, "w") as fp:
            json.dump(train_stats, fp, indent=4)
    else:
        print(f"Loading stats from: {mean_std_file=}")
        with open(mean_std_file, "r") as fp:
            train_stats = json.load(fp)
            
    aos_mean = train_stats["AOS_mean"]
    aos_std = train_stats["AOS_std"]
    res_mean = train_stats["RES_mean"]
    res_std = train_stats["RES_std"]
    
    dataloader = DataLoader(
        train_ds,
        batch_size=conf.train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=conf.num_workers,
    )
    
    val_dataloader = DataLoader(
        test_ds,
        batch_size=conf.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=conf.num_workers // 2,
    )
    
    if conf.model.arch == "mambaout":
        num_class_embeds = None if conf.model.num_cond_embed <= 0 else conf.model.num_cond_embed
        model = MambaOutIR(
            1,
            (96, 192, 384), 
            (2, 8, 2),
            num_class_embeds,
            oss_refine_blocks=2,
            local_embeds=False,
            drop_path=0.2,
            with_stem=True,
        )
    else:
        raise NotImplementedError("Future work.")
    
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
    
    if conf.predict_image:
        criterion = ImageCriterion(
            aos_mean=aos_mean,
            aos_std=aos_std,
            use_ssim=conf.use_ssim,
            objective=conf.objective,
            resnet_type=conf.resnet_type,
            up_to_layer=conf.up_to_layer,
            perc_weight=conf.perc_weight,
            l1_weight=conf.l1_weight,
        )
        val_criterion = ImageCriterion(
            aos_mean=aos_mean,
            aos_std=aos_std,
            use_ssim=True,
        ).to(accelerator.device)
    else:
        raise NotImplementedError("Future work.")
    
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
                
                if conf.data.img_sz is not None:
                    img_sz = conf.data.img_sz
                    aos = F.interpolate(aos, size=(img_sz, img_sz))  # Bx1xHxW
                    gt = F.interpolate(gt, size=(img_sz, img_sz))  # Bx1xHxW
                
                aos_normalized = (aos - aos_mean) / aos_std
                
                optimizer.zero_grad()
                
                # Working with env. temp. from 0 to Tmax (int) as conditions
                # if isinstance(model, (VmambaIR, MambaOutIR)):
                if isinstance(model, MambaOutIR):
                    if model.local_embeds:  # Local embedding
                        pred_res_or_img = model(aos_normalized, None, et, conf.add_residual)
                    elif model.embedding is not None:  # Global embedding
                        pred_res_or_img = model(aos_normalized, et, None, conf.add_residual)
                    else:  # No embedding
                        pred_res_or_img = model(aos_normalized, None, None, conf.add_residual)
                else:  # Other models
                    if model.embedding is not None:
                        pred_res_or_img = model(aos_normalized, et, conf.add_residual)
                    else:
                        pred_res_or_img = model(aos_normalized, None, conf.add_residual)
                        
                loss = criterion(pred_res_or_img, aos, gt)
                if not isinstance(loss, torch.Tensor):
                    loss_dict = loss  # Actually a dictionary.
                    loss = loss_dict["total_loss"]
                    
                    if global_step % conf.log_every == 0:
                        writer.add_scalar("train/total_loss", loss.item(), global_step)
                        if conf.objective == "PERC":
                            l1_loss = loss_dict["l1_loss"]
                            perc_loss = loss_dict["perc_loss"]
                            writer.add_scalar("train/l1_loss", l1_loss, global_step)
                            writer.add_scalar("train/perc_loss", perc_loss, global_step)
                            
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                
                total_loss += loss.item()

                pred = to_image(pred_res_or_img.detach(), aos, aos_std, aos_mean, conf, res_std, res_mean)
                metrics.update(pred, gt)

                if global_step % conf.log_every == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    writer.add_scalar("train/lr", lr, global_step)
                    pbar.set_postfix({"total_loss": f"{loss.item():.3e}"})
                
                if global_step % conf.vis_every == 0:
                    log_images(
                        writer,
                        global_step,
                        "train",
                        pred_res_or_img,
                        aos,
                        aos_std,
                        aos_mean,
                        res_std,
                        res_mean,
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
                        
                        if conf.data.img_sz is not None:
                            img_sz = conf.data.img_sz
                            aos = F.interpolate(aos, size=(img_sz, img_sz))  # Bx1xHxW
                            gt = F.interpolate(gt, size=(img_sz, img_sz))  # Bx1xHxW

                        aos_normalized = (aos - aos_mean) / aos_std

                        # Working with env. temp. from 0 to Tmax (int) as conditions
                        # if isinstance(model, (VmambaIR, MambaOutIR)):
                        if isinstance(model, MambaOutIR):
                            if model.local_embeds:  # Local embedding
                                pred_res_or_img = model(aos_normalized, None, et, conf.add_residual)
                            elif model.embedding is not None:  # Global embedding
                                pred_res_or_img = model(aos_normalized, et, None, conf.add_residual)
                            else:  # No embedding
                                pred_res_or_img = model(aos_normalized, None, None, conf.add_residual)
                        else:  # Other models
                            if model.embedding is not None:
                                pred_res_or_img = model(aos_normalized, et, conf.add_residual)
                            else:
                                pred_res_or_img = model(aos_normalized, None, conf.add_residual)

                        loss = val_criterion(pred_res_or_img, aos, gt)
                        if not isinstance(loss, torch.Tensor):
                            total_loss = loss["total_loss"]
                        val_loss += total_loss.item()

                        pred = to_image(pred_res_or_img, aos, aos_std, aos_mean, conf, res_std, res_mean)
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
                        folders.append(test_ds.root / test_ds.folders[i])
                    
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
                    res_std,
                    res_mean,
                    gt,
                    conf,
                )

                sys.stdout.flush()
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            epoch += 1

        writer.close()

    train_loop()