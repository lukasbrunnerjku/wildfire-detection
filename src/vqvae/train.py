from dataclasses import dataclass, field
from typing import Optional
from omegaconf import OmegaConf
from enum import Enum
from multiprocessing import cpu_count
from diffusers import AutoencoderKL, VQModel
import math
import torch
from torch import Tensor
from PIL import Image
import numpy as np
import torch.nn.functional as F
import gc
from torchmetrics import MeanSquaredError
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from .perceptual_loss import PerceptualLoss
from .gan import AdvConfig, AdversarialLoss
from .quantizers import (
    CodebookSize,
    Quantizer,
    QuantizerType,
    add_histogram_raw,
    perplexity,
)
from ..utils.conf import DataConfig
from ..utils.torch_conf import manual_seed_all, set_torch_options
from ..utils.weight_decay import adjusted_weight_decay, weight_decay_parameter_split
from ..utils.fid import FID, build_fid_metric
from ..utils.image import (
    normalize_tif,
    denormalize_tif,
    to_zero_one_range,
    pil_make_grid,
    tone_mapping,
)
from ..utils.data import DataModule
from ..utils.lit_ema import EMACallback


class AutoEncoderType(Enum):
    AutoencoderKL = 0
    VQModel = 1


@dataclass
class TrainConfig:
    logdir: Optional[str] = None
    ae_type: AutoEncoderType = AutoEncoderType.VQModel
    quantizer_type: QuantizerType = QuantizerType.LFQ
    codebook_size: CodebookSize = CodebookSize.MEDIUM
    # Some quantizers ignore latent_dim, calculating it from the codebook_size.
    latent_dim: int = 3
    train_batch_size: int = 16
    eval_batch_size: int = 128
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 500
    train_steps: int = 19500
    seed: int = 42
    device: str = "cuda"
    num_workers: int = -1
    log_every_n_steps: int = 50
    vis_every_n_steps: int = 500
    val_every_n_steps: int = 500
    gradient_clip_val: float = 1.0
    # commit_weight has no effect on LFQ (using default) or FSQ (has no auxillary loss).
    commit_weight: float = 1e-2
    kl_weight: float = 1e-6
    l1_weight: float = 0.1
    l2_weight: float = 1.0
    perc_weight: float = 0.5
    resnet_type: str = "resnet18"
    up_to_layer: Optional[int] = 2
    use_ema: bool = False
    ema_decay: float = 0.995
    ema_update_every: int = 10
    ema_update_after_step: int = 0
    adv: AdvConfig = field(default_factory=AdvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    fast_dev_run: bool = False

    def __post_init__(self):
        if self.num_workers == -1:
            self.num_workers = cpu_count()

        if self.fast_dev_run:
            self.num_workers = 0
            self.num_sanity_steps = 2


class VQVAE(LightningModule):

    def __init__(self, config=None):
        super().__init__()

        if config.ae_type == AutoEncoderType.AutoencoderKL:
            self.ae = AutoencoderKL(
                in_channels=1,
                out_channels=1,
                latent_channels=config.latent_dim,
                sample_size=config.data.image_size,
                block_out_channels=(64, 128, 256),
                mid_block_add_attention=False,
                down_block_types=(
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                ),
                up_block_types=(
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                ),
            )
        elif config.ae_type == AutoEncoderType.VQModel:
            base_two_exponent = int(math.log2(config.codebook_size.value))
            quantize = Quantizer(config.quantizer_type, base_two_exponent)
            config.latent_dim = quantize.quantizer.codebook_dim

            # In case of FSQ it is approximately the given codebook size.
            self.codebook_size = quantize.quantizer.codebook_size

            if config.quantizer_type == QuantizerType.LFQ:
                # Proper weighting already done inside the quantizer.
                config.commit_weight = 1.0

            self.ae = VQModel(
                in_channels=1,
                out_channels=1,
                num_vq_embeddings=self.codebook_size,
                vq_embed_dim=config.latent_dim,
                latent_channels=config.latent_dim,
                block_out_channels=(64, 128, 256),
                mid_block_add_attention=False,
                down_block_types=(
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                ),
                up_block_types=(
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                ),
            )
        
        self.perceptual_loss = PerceptualLoss(config.resnet_type, config.up_to_layer)
        
        if config.adv.start_step >= 0:
            self.adversarial_loss = AdversarialLoss(
                in_channels=1,
                disc_weight=config.adv.disc_weight,
                start_step=config.adv.start_step,
                adaptive_weight=config.adv.adaptive_weight,
                loss_type=config.adv.loss_type,
                lecam_reg_weight=config.adv.lecam_reg_weight,
                ema_decay=config.adv.ema_decay,
                n_layers=config.adv.n_layers,
            )

        self.mse = MeanSquaredError()
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.fid: FID = build_fid_metric(image_size=config.data.image_size)

        self.config = config
        self.save_hyperparameters(config)
        self.first_validation_epoch_end = True
        self.validation_step_hist = torch.tensor(0, dtype=torch.long)

    def configure_optimizers(self):
        weight_decay = adjusted_weight_decay(
            self.config.weight_decay,
            self.config.train_batch_size,
            self.config.train_steps
        )

        decay, no_decay = weight_decay_parameter_split(self.ae)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        ae_optimizer = torch.optim.AdamW(
            groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )
        ae_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            ae_optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    ae_optimizer, 1e-2, total_iters=self.config.warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    ae_optimizer,
                    T_max=self.config.train_steps,
                    eta_min=self.config.min_learning_rate,
                ),
            ],
            milestones=[self.config.warmup_steps],
        )
        ae_dict = {
            "optimizer": ae_optimizer,
            "lr_scheduler": {
                "scheduler": ae_lr_scheduler,
                "interval": "step",        
            },
        }
        
        self.automatic_optimization = False
        self.manual_global_step = 0

        if self.config.adv.start_step >= 0:
            decay, no_decay = weight_decay_parameter_split(self.adversarial_loss)
            groups = [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ]
            disc_optimizer = torch.optim.AdamW(
                groups,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
            )
            disc_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                disc_optimizer,
                [
                    torch.optim.lr_scheduler.LinearLR(
                        disc_optimizer, 1e-2, total_iters=self.config.warmup_steps
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        disc_optimizer,
                        T_max=self.config.train_steps,
                        eta_min=self.config.min_learning_rate,
                    ),
                ],
                milestones=[self.config.warmup_steps],
            )
            disc_dict = {
                "optimizer": disc_optimizer,
                "lr_scheduler": {
                    "scheduler": disc_lr_scheduler,
                    "interval": "step",        
                },
            }
            return (ae_dict, disc_dict)
        
        else:
            return ae_dict
    
    def training_step(self, batch, batch_idx) -> Optional[Tensor]:
        x = batch  # BxCxHxW; float32; on target device
        x = normalize_tif(x, self.config.data.mean, self.config.data.std)  # ~N(0, 1)

        if isinstance(self.ae, VQModel):
            output = self.ae(x)
            commit_loss = output.commit_loss
            x_hat = output.sample
        else:
            posterior = self.ae.encode(x).latent_dist
            z = posterior.sample()  # Reparametrization trick.
            x_hat = self.ae.decode(z).sample
            kl_loss = posterior.kl().mean()

        if self.manual_global_step % self.config.vis_every_n_steps == 0:
            if self.global_rank == 0:
                subset_x = denormalize_tif(x[:4].cpu(), self.config.data.mean, self.config.data.std)
                subset_x_hat = denormalize_tif(x_hat[:4].detach().cpu(), self.config.data.mean, self.config.data.std)
                imgs = (
                    [tone_mapping(img, img.min(), img.max()) for img in subset_x] +
                    [tone_mapping(img, img.min(), img.max()) for img in subset_x_hat]
                )
                grid: Image.Image = pil_make_grid(imgs, ncol=4)
                self.logger.experiment.add_image(
                    "sample_reconstruction",
                    np.asarray(grid),
                    self.manual_global_step,
                    dataformats="HWC",
                )

        perc_loss = self.perceptual_loss(x, x_hat)
        l1_loss = F.l1_loss(x, x_hat, reduction="mean")
        l2_loss = F.mse_loss(x, x_hat, reduction="mean")
        loss = (
            self.config.perc_weight * perc_loss +
            self.config.l1_weight * l1_loss +
            self.config.l2_weight * l2_loss
        )

        if isinstance(self.ae, VQModel):
            loss = loss + self.config.commit_weight * commit_loss
        else:
            loss = loss + self.config.kl_weight * kl_loss

        if self.config.adv.start_step >= 0:
            ae_opt, disc_opt = self.optimizers()

            ae_opt.zero_grad()
            g_loss, g_weight = self.adversarial_loss.forward_autoencoder(
                # The perception loss is for adaptive weighting to ensure the
                # gradient of the perception loss has equal contribution as the
                # gradient from the adversarial loss (magnitude).
                perc_loss,  
                x_hat,
                self.manual_global_step,
                self.ae.decoder.conv_out.weight,
            )
            loss = loss + g_weight * g_loss
            self.manual_backward(loss)
            ae_opt.step()
            
            loss, d_loss, lecam_penalty = self.adversarial_loss.forward_discriminator(
                x, x_hat, self.manual_global_step, True,
            )

            if loss is not None:  # Depends on "config.adv.start_step"
                disc_opt.zero_grad()
                self.manual_backward(loss)
                disc_opt.step()

            if isinstance(self.ae, VQModel):
                self.log("vqvae_commit_loss", commit_loss.detach().cpu().item())
            else:
                self.log("vqvae_kl_loss", kl_loss.detach().cpu().item())

            self.log("vqvae_perc_loss", perc_loss.detach().cpu().item())
            self.log("vqvae_l1_loss", l1_loss.detach().cpu().item())
            self.log("vqvae_l2_loss", l2_loss.detach().cpu().item())
            self.log("vqvae_lecam_penalty", lecam_penalty.detach().cpu().item())
            self.log("vqvae_d_loss", d_loss.detach().cpu().item())
            self.log("vqvae_g_loss", g_loss.detach().cpu().item())
            self.log("vqvae_adv_logits_real", self.adversarial_loss.last_logits_real.cpu().item())
            self.log("vqvae_adv_logits_fake", self.adversarial_loss.last_logits_fake.cpu().item())

            ae_sc, disc_sc = self.lr_schedulers()
            ae_sc.step()
            disc_sc.step()

        else:
            ae_opt = self.optimizers()
            ae_opt.zero_grad()
            self.manual_backward(loss)
            ae_opt.step()

            if isinstance(self.ae, VQModel):
                self.log("vqvae_commit_loss", commit_loss.detach().cpu().item())
            else:
                self.log("vqvae_kl_loss", kl_loss.detach().cpu().item())

            self.log("vqvae_perc_loss", perc_loss.detach().cpu().item())
            self.log("vqvae_l1_loss", l1_loss.detach().cpu().item())
            self.log("vqvae_l2_loss", l2_loss.detach().cpu().item())

            ae_sc = self.lr_schedulers()
            ae_sc.step()
        
        self.manual_global_step += 1

    def on_validation_epoch_start(self):
        gc.collect()
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        x_float: Tensor = batch  # BxCxHxW; float32; on target device
        x_float_norm = normalize_tif(x_float, self.config.data.mean, self.config.data.std)  # BxCxHxW; ~N(0, 1)

        if isinstance(self.ae, VQModel):
            # Reconstruction, from VQModel but with access to indices.
            h = self.ae.encode(x_float_norm).latents
            quant, _, (_, _, code_indices) = self.ae.quantize(h)
            quant2 = self.ae.post_quant_conv(quant)
            x_hat_float_norm = self.ae.decoder(
                quant2, quant if self.ae.config.norm_type == "spatial" else None
            )  # BxCxHxW
            if self.global_rank == 0:
                code_size = self.codebook_size  # Might be different than config.codebook_size!
                code_counts = torch.bincount(code_indices, minlength=code_size)
                self.validation_step_hist = self.validation_step_hist + code_counts.cpu()
        else:
            x_hat_float_norm = self.ae(x_float_norm).sample  # BxCxHxW

        x_hat_float = denormalize_tif(x_hat_float_norm, self.config.data.mean, self.config.data.std)  # BxCxHxW

        if batch_idx == 0 and self.global_rank == 0:
            subset_x = x_float[:4].cpu()
            subset_x_hat = x_hat_float[:4].cpu()
            imgs = (
                [tone_mapping(img, img.min(), img.max()) for img in subset_x] +
                [tone_mapping(img, img.min(), img.max()) for img in subset_x_hat]
            )
            grid: Image.Image = pil_make_grid(imgs, ncol=4)
            self.logger.experiment.add_image(
                "mode_reconstruction",
                np.asarray(grid),
                self.manual_global_step,
                dataformats="HWC",
            )

        # Keep data range of training to be compatible with the training loss.
        self.mse.update(x_hat_float_norm, x_float_norm)  # BxCxHxW

        with torch.autocast("cuda", enabled=False):
            # Typically calculated on [0., 1.] data range.
            x_hat_float = to_zero_one_range(x_hat_float, x_hat_float.min(), x_hat_float.max())
            x_float = to_zero_one_range(x_float, x_float.min(), x_float.max())
            self.psnr.update(x_hat_float, x_float)  # BxCxHxW; [0., 1.]
            self.ssim.update(x_hat_float, x_float)  # BxCxHxW; [0., 1.]

            # Expects Bx3xHxW; unnormalized!
            if self.fid.reset_real_features or self.first_validation_epoch_end:
                self.fid.update(x_float.repeat(1, 3, 1, 1), real=True)
            
            self.fid.update(x_hat_float.repeat(1, 3, 1, 1), real=False)

    def on_validation_epoch_end(self):
        mse_score = self.mse.compute()
        psnr_score = self.psnr.compute()
        ssim_score = self.ssim.compute()

        real_num_samples = self.fid.real_features_num_samples.cpu().item()
        fake_num_samples = self.fid.fake_features_num_samples.cpu().item()
        self.print(f"on_validation_epoch_end >> {real_num_samples=} {fake_num_samples=}")

        with torch.autocast("cuda", enabled=False):
            fid_score = self.fid.compute()

        if self.global_rank == 0 and self.first_validation_epoch_end:
            # Initialize hyperparameter tab in tensorboard with initial value.
            self.first_validation_epoch_end = False
            self.logger.log_hyperparams(self.hparams, metrics={"fid_score": fid_score})
        
        # For torchmetrics use sync_dist=False, handled automatically. 
        # https://www.restack.io/p/pytorch-lightning-answer-sync-dist-cat-ai
        self.log("vqvae_mse_score", mse_score)
        self.log("vqvae_psnr_score", psnr_score)
        self.log("vqvae_ssim_score", ssim_score)
        self.log("fid_score", fid_score)

        self.mse.reset()
        self.psnr.reset()
        self.ssim.reset()
        # Might reset only fake features. See "reset_real_features" attribute.
        self.fid.reset()

        if self.global_rank == 0 and isinstance(self.ae, VQModel):
            hist = self.validation_step_hist
            add_histogram_raw(
                self.logger.experiment, "vqvae_codebook_hist", hist, self.manual_global_step
            )
            pp = perplexity(hist)
            pp = (pp - 1.0) / (len(hist) - 1)  # norm to [0, 1]
            self.logger.experiment.add_scalar(
                "vqvae_codebook_perplexity", pp, self.manual_global_step
            )
            self.validation_step_hist = torch.tensor(0, dtype=torch.long)

        gc.collect()
        torch.cuda.empty_cache()
    

def main():
    conf = OmegaConf.structured(TrainConfig())
    args = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, args)

    assert conf.logdir is not None
    assert conf.data.folder is not None
    assert conf.data.mean is not None
    assert conf.data.std is not None

    print(conf)

    manual_seed_all(conf.seed)
    set_torch_options()

    datamodule = DataModule(conf)
    model = VQVAE(conf)

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
        callbacks.append(EMACallback(
            model=model.ae,
            decay=conf.ema_decay,
            update_every=conf.ema_update_every,
            update_after_step=conf.ema_update_after_step,
        ))

    max_steps = (conf.warmup_steps + conf.train_steps)

    if conf.adv.start_step >= 0:
        max_steps *= 2

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
        num_sanity_val_steps=-1,  
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
