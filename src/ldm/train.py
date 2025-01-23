from dataclasses import dataclass, field
from typing import Optional
from omegaconf import OmegaConf
from multiprocessing import cpu_count

from ..utils.conf import DataConfig

# mean, std of data: mean: float = 2.7935, std: float = 5.9023

class LitLDM(LightningModule):

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        self.vqvae: VQModel = load_ae_from_checkpoint_(None, config.load_ae_checkpoint)
        stride = 2**(len(self.vqvae.encoder.down_blocks) - 1)
        latent_dim: int = self.vqvae.config.vq_embed_dim
    
        self.unet_config = dict(
            sample_size=int(config.image_size / stride),
            in_channels=latent_dim,
            out_channels=latent_dim,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        self.unet = UNet2DModel(**self.unet_config)
        self.restarted_from_ckpt = False

        # From "ptx0/pseudo-journey-v2" Huggingface model
        # See stable_diffusion.py --> pipe.scheduler.config
        stable_diffusion_kwargs = {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "clip_sample": False,
            "set_alpha_to_one": False,
            "steps_offset": 1,
            # https://arxiv.org/pdf/2305.08891
            # When SNR is zero, ϵ prediction becomes a trivial task and ϵ
            # loss cannot guide the model to learn anything meaningful about
            # the data, thus if prediction_type is "epsilon" set
            # "rescale_betas_zero_snr" to False!
            # (--> see 3.2. Train with V Prediction and V Loss)
            "prediction_type": "v_prediction",
            "thresholding": False,
            "dynamic_thresholding_ratio": 0.995,
            "clip_sample_range": 1.0,
            "sample_max_value": 1.0,
            # https://arxiv.org/pdf/2305.08891
            # (--> see 3.3. Sample from the Last Timestep)
            "timestep_spacing": "trailing",
            # https://arxiv.org/pdf/2305.08891
            # SNR(T) is otherwise in standard implementations not zero
            # thus it would result in discrepancy between training/inference
            # because in training the lowest frequencies would be kept while 
            # during inference we sample from pure noise. Note that T is the 
            # timestep at which we should have pure noise.
            # (--> see 3.1 Enforce Zero Terminal SNR)
            "rescale_betas_zero_snr": True,
        }
        self.noise_scheduler = DDIMScheduler(**stable_diffusion_kwargs)
        self.pipeline = LDMPipeline(
            vqvae=self.vqvae,
            unet=self.unet,
            scheduler=self.noise_scheduler,
        )

        self.fid: CustomFID = build_custom_fid_metric(image_size=config.image_size)

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
        if self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            x = batch  # BxCxHxW; float32; on target device
            x = normalize_tif(x)  # ~N(0, 1)
            
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
        x = normalize_tif(x)  # ~N(0, 1)

        if self.global_rank == 0 and self.global_step == 0:
            # Visualize an example batch of real-world images.
            subset_x = denormalize_tif(x[:9].cpu())
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
        
        # NOTE: With "v_prediction" the model_output is not the noise epsilon, but the velocity.
        # https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
        noise_pred = self.unet(noisy_h, timesteps, return_dict=False)[0]

        if self.config.snr_gamma == 0.0:
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
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
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        # Sample some images from random noise (this is the backward diffusion process).
        x_hat_float: np.ndarray = self.pipeline(
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





@dataclass
class TrainConfig:
    vqvae_checkpoint: Optional[str] = None
    train_batch_size: int = 16
    eval_batch_size: int = 128
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
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
    fast_dev_run: bool = True

    def __post_init__(self):
        if self.num_workers == -1:
            self.num_workers = cpu_count()

        if self.fast_dev_run:
            self.num_workers = 0
            self.num_sanity_steps = 2


def main():
    conf = OmegaConf.structured(TrainConfig())
    args = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, args)

    assert conf.vqvae_checkpoint is not None
    assert conf.data.folder is not None

    print(conf)


if __name__ == "__main__":
    """
    
    """
    main()
