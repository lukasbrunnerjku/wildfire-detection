from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from omegaconf import OmegaConf
from multiprocessing import cpu_count
import torch
from torch import Tensor
from PIL import Image
import numpy as np
import torch.nn.functional as F
from diffusers import VQModel, DDIMScheduler
from diffusers.training_utils import EMAModel, compute_snr
import gc
import logging
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from .model import UNet2DModel, LDMPipeline
from ..utils.conf import DataConfig
from ..utils.lit_logging import setup_logger, log_main_process
from ..utils.torch_conf import manual_seed_all, set_torch_options
from ..utils.weight_decay import adjusted_weight_decay, weight_decay_parameter_split
from ..utils.fid import FID, build_fid_metric
from ..utils.image import (
    normalize_tif,
    denormalize_tif,
    pil_make_grid,
    tone_mapping,
)
from ..utils.data import DataModule
from ..vqvae.train import load_ae_from_checkpoint, AutoEncoderType


@dataclass
class TrainConfig:
    logdir: Optional[str] = None
    vqvae_checkpoint: Optional[str] = None
    block_out_channels: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    train_batch_size: int = 16
    eval_batch_size: int = 128
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    warmup_steps: int = 500
    train_steps: int = 19500
    seed: int = 42
    device: str = "cuda"
    num_workers: int = -1
    log_every_n_steps: int = 50
    vis_every_n_steps: int = 500 
    val_every_n_steps: int = 500
    num_sanity_steps: int = -1
    gradient_clip_val: float = 1.0
    use_ema: bool = True
    ema_power: float = 0.75
    ema_max_decay: float = 0.995
    snr_gamma: float = 5.0
    data: DataConfig = field(default_factory=DataConfig)
    fast_dev_run: bool = False
    
    def __post_init__(self):
        if self.num_workers == -1:
            self.num_workers = cpu_count()

        if self.fast_dev_run:
            self.num_workers = 0
            self.num_sanity_steps = 2


class LDM(LightningModule):

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        self.restarted_from_ckpt = False
        
        self.vqvae: VQModel = load_ae_from_checkpoint(config.vqvae_checkpoint)
        stride = 2**(len(self.vqvae.encoder.down_blocks) - 1)
        latent_dim: int = self.vqvae.config.vq_embed_dim

        n_unet_blocks = len(config.block_out_channels)
        self.unet = UNet2DModel(
            sample_size=int(config.data.image_size / stride),
            in_channels=latent_dim,
            out_channels=latent_dim,
            layers_per_block=2,
            block_out_channels=config.block_out_channels,
            down_block_types=tuple("DownBlock2D" for _ in range(n_unet_blocks)),
            up_block_types=tuple("UpBlock2D" for _ in range(n_unet_blocks)),
        )
        
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="v_prediction",
            thresholding=False,
            dynamic_thresholding_ratio=0.995,
            clip_sample_range=1.0,
            sample_max_value=1.0,
            timestep_spacing="trailing",
            rescale_betas_zero_snr=True,
        )  # https://arxiv.org/pdf/2305.08891

        self.pipeline = LDMPipeline(
            vqvae=self.vqvae,
            unet=self.unet,
            scheduler=self.noise_scheduler,
        )

        self.fid: FID = build_fid_metric(image_size=config.data.image_size)

        self.config = config
        self.first_validation_epoch_end = True

    def configure_optimizers(self):
        weight_decay = adjusted_weight_decay(
            self.config.weight_decay,
            self.config.train_batch_size,
            self.config.train_steps
        )
        decay, no_decay = weight_decay_parameter_split(self.unet)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(
            groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, 1e-2, total_iters=self.config.warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.train_steps,
                    eta_min=self.config.min_learning_rate,
                ),
            ],
            milestones=[self.config.warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",        
            },
        }
    
    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        if (
            self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
        ):
            x = batch  # BxCxHxW; float32; on target device
            x = normalize_tif(x, self.config.data.mean, self.config.data.std)  # ~N(0, 1)
            
            with torch.inference_mode():
                h = self.vqvae.encode(x).latents  # BxCxHxW
                h = h.flatten()

            latent_mean = h.mean()
            latent_std = h.std()
            self.print(f"{latent_mean=} {latent_std=}")

            self.unet.latent_mean.data = latent_mean
            self.unet.latent_std.data = latent_std

    @torch.inference_mode()
    def generate_new_images(
        self,
        batch_size: int = 9,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
    ) -> list[Image.Image]:
        # Sample some images from random noise (this is the backward diffusion process).
        images: list[Image.Image] = self.pipeline(
            self.config.data.mean,
            self.config.data.std,
            batch_size=batch_size,
            generator=generator,
            output_type=output_type,
        ).images
        return images

    @rank_zero_only
    @torch.inference_mode()
    def demo_images(self) -> Tensor:
        # Use a separate torch generator to avoid rewinding the random state of the main training loop
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        images = self.generate_new_images(batch_size=9, generator=generator)
        grid: Image.Image = pil_make_grid(images, ncol=3)
        return grid
    
    def training_step(self, batch, batch_idx) -> Tensor:
        x = batch  # BxCxHxW; float32; on target device
        x = normalize_tif(x, self.config.data.mean, self.config.data.std)  # ~N(0, 1)

        if self.global_rank == 0 and self.global_step == 0:
            # Visualize an example batch of real-world images.
            subset_x = denormalize_tif(x[:9].cpu(), self.config.data.mean, self.config.data.std)
            imgs = [tone_mapping(img, img.min(), img.max()) for img in subset_x]
            grid: Image.Image = pil_make_grid(imgs, ncol=3)
            self.logger.experiment.add_image(
                "real_images",
                np.asarray(grid),
                self.global_step,
                dataformats="HWC",
            )

        with torch.inference_mode():
            # Images would also be scaled to zero mean, unit variance.
            h = self.vqvae.encode(x).latents  # BxCxHxW
            h = (h - self.unet.latent_mean) / self.unet.latent_std 
            B = h.shape[0]

        noise = torch.randn(h.shape, device=h.device, dtype=h.dtype)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            size=(B,),
            device=x.device,
            dtype=torch.int64
        )
        noisy_h = self.noise_scheduler.add_noise(h, noise, timesteps)

        target = self.noise_scheduler.get_velocity(h, noise, timesteps)
        
        noise_pred = self.unet(noisy_h, timesteps, return_dict=False)[0]

        if self.config.snr_gamma == 0.0:
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        else:
            snr = compute_snr(self.noise_scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, self.config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                dim=1
            )[0]
            mse_loss_weights = mse_loss_weights / (snr + 1)
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        
        self.log("diffusion_loss", loss, prog_bar=True)

        if (self.global_step % self.config.vis_every_n_steps == 0) and self.global_rank == 0:
            grid = self.demo_images()
            self.logger.experiment.add_image(
                "demo_images",
                np.asarray(grid),
                self.global_step,
                dataformats="HWC",
            )

        return loss
    
    def on_validation_epoch_start(self):
        gc.collect()
        torch.cuda.empty_cache()
    
    def validation_step(self, batch, batch_idx):
        # https://huggingface.co/docs/diffusers/conceptual/evaluation
        x_float: Tensor = batch  # BxCxHxW; float32; on target device

        # Use a separate torch generator to avoid rewinding the random state of the main training loop
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        # Sample some images from random noise (this is the backward diffusion process).
        x_hat_float: np.ndarray = self.pipeline(
            self.config.data.mean,
            self.config.data.std,
            batch_size=x_float.shape[0],
            generator=generator,
            output_type="raw",
        ).images  # Bx1xHxW; float32

        with torch.autocast("cuda", enabled=False):
            # Bx3xHxW; float32; unnormalized expected!
            if self.fid.reset_real_features or self.first_validation_epoch_end:
                self.fid.update(x_float.repeat(1, 3, 1, 1), real=True)

            self.fid.update(x_hat_float.repeat(1, 3, 1, 1), real=False)

    def on_validation_epoch_end(self):
        real_num_samples = self.fid.real_features_num_samples.cpu().item()
        fake_num_samples = self.fid.fake_features_num_samples.cpu().item()
        self.print(f"on_validation_epoch_end >> {real_num_samples=} {fake_num_samples=}")

        with torch.autocast("cuda", enabled=False):
            fid_score = self.fid.compute()

        if self.global_rank == 0 and self.first_validation_epoch_end:
            # Initialize hyperparameter tab in tensorboard with initial value.
            self.first_validation_epoch_end = False
            self.logger.log_hyperparams(self.hparams, metrics={"fid_score": fid_score})
        
        self.log("fid_score", fid_score)

        # Might reset only fake features. See "reset_real_features" attribute.
        self.fid.reset()

        gc.collect()
        torch.cuda.empty_cache()


class EMACallbackUNet(Callback):
    """
    Model Exponential Moving Average. Empirically it has been found that using the moving average
    of the trained parameters of a deep network is better than using its trained parameters directly.

    The ema parameters of the network is set after training end.

    Adapted from lightning callback
    https://github.com/benihime91/gale/blob/master/gale/collections/callbacks/ema.py
    """

    def __init__(self):
        self.ema = None
        self.logger = None

    def on_fit_start(self, trainer, pl_module: LDM):
        config = pl_module.hparams
        self.ema = EMAModel(
            pl_module.unet.parameters(),
            decay=config.ema_max_decay,
            use_ema_warmup=True,
            power=config.ema_power,
            model_cls=UNet2DModel,
            model_config=pl_module.unet.config,
        )
        self.logger = setup_logger(trainer.global_rank)

    def on_train_batch_end(
        self, trainer, pl_module: LDM, outputs, batch, batch_idx,
    ):
        # Update currently maintained parameters.
        self.ema.step(pl_module.unet.parameters())

    def on_validation_epoch_start(self, trainer, pl_module: LDM):
        # save original parameters before replacing with EMA version
        self.ema.store(pl_module.unet.parameters())

        # copy EMA parameters to UNet model
        self.ema.copy_to(pl_module.unet.parameters())

        log_main_process(
            self.logger,
            logging.INFO,
            "Using EMA weights for UNet in validation loop.",
        )
        
    def on_validation_end(self, trainer, pl_module: LDM):
        "Restore original parameters to resume training later"
        self.ema.restore(pl_module.unet.parameters())

    def on_train_end(self, trainer, pl_module: LDM):
        # copy EMA parameters to UNet model
        self.ema.copy_to(pl_module.unet.parameters())
            
    def on_save_checkpoint(self, trainer, pl_module: LDM, checkpoint: Dict[str, Any]) -> None:
        # checkpoint: the checkpoint dictionary that will be saved.
        checkpoint["state_dict_ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer, pl_module: LDM, checkpoint: Dict[str, Any]) -> None:
        # https://github.com/zyinghua/uncond-image-generation-ldm/blob/main/train.py#L339
        self.ema.load_state_dict(checkpoint["state_dict_ema"])

        # copy EMA parameters to UNet model
        self.ema.copy_to(pl_module.unet.parameters())


def load_shadow_params(checkpoint: str):
    ckpt = torch.load(checkpoint, weights_only=False)
    shadow_params = ckpt["state_dict_ema"]["shadow_params"]
    return shadow_params


def copy_to(shadow_params, parameters):
    # shadow_params --> parameters
    for s_param, param in zip(shadow_params, parameters):
        param.data.copy_(s_param.to(param.device).data)


def load_ldm_from_checkpoint(ckpt: str, use_ema_weights: bool = True) -> LDM:
    model = LDM.load_from_checkpoint(ckpt)
    if use_ema_weights:
        shadow_params = load_shadow_params(ckpt)
        copy_to(shadow_params, model.unet.parameters())
    return model


def main():
    conf = OmegaConf.structured(TrainConfig())
    file = OmegaConf.load("src/configs/ldm.yaml")
    args = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, file, args)

    assert conf.vqvae_checkpoint is not None
    assert conf.logdir is not None
    assert conf.data.folder is not None
    assert conf.data.mean is not None
    assert conf.data.std is not None

    print(conf)

    manual_seed_all(conf.seed)
    set_torch_options()

    datamodule = DataModule(conf)
    model = LDM(conf)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(
        filename="{step:03d}-{fid_score:.3f}",
        save_top_k=1,
        monitor="fid_score",
        mode="min",  # Lower FID is better!
        save_last=True,
        save_weights_only=False,
    )

    callbacks = [lr_monitor, model_checkpoint]

    if conf.use_ema:
        callbacks.append(EMACallbackUNet())

    max_steps = (conf.warmup_steps + conf.train_steps)

    trainer = Trainer(
        fast_dev_run=4 * int(conf.fast_dev_run),
        accelerator=conf.device,
        devices="0,",
        strategy="auto",
        num_nodes=1,
        log_every_n_steps=conf.log_every_n_steps,
        max_steps=max_steps,
        val_check_interval=conf.val_every_n_steps,
        precision="16-mixed",
        logger=TensorBoardLogger(
            conf.logdir,
            name=model.__class__.__name__,
            default_hp_metric=False,
        ),
        callbacks=callbacks,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=conf.num_sanity_steps,  
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    # python -m src.ldm.train data.folder=/mnt/data/wildfire/imgs1 data.mean=2.7935 data.std=5.9023 logdir=/mnt/data/wildfire/runs vqvae_checkpoint=/mnt/data/wildfire/runs/VQVAE/version_1/checkpoints/last.ckpt
    main()
