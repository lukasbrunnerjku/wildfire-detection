import torch
from torch import Tensor
from torch.utils.data import DataLoader
from pathlib import Path
from diffusers import VQModel
from PIL import Image
from tqdm import tqdm
import numpy as np

from ..utils.image import (
    normalize_tif,
    denormalize_tif,
    tone_mapping,
)
from ..utils.image import TemperatureDataWithIndices
from ..vqvae.train import load_ae_from_checkpoint, TrainConfig, AutoEncoderType
from ..utils.torch_conf import set_torch_options


@torch.inference_mode()
def encode(vqvae: VQModel, x: Tensor, mean: float, std: float) -> Tensor:
    # x ... BxCxHxW; float32; on target device
    x = normalize_tif(x, config.mean, config.std)  # ~N(0, 1)
    latents = vqvae.encode(x).latents  # BxCxHxW
    return latents


@torch.inference_mode()
def decode(vqvae: VQModel, latents: Tensor, mean: float, std: float) -> Tensor:
    # latents ... BxCxHxW; float32; on target device
    x = vqvae.decode(latents)["sample"]
    x = denormalize_tif(x, mean, std)
    return x


if __name__ == "__main__":
    # python -m src.scripts.vqvae_compression
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder", default="/mnt/data/wildfire/imgs1")
    parser.add_argument("--vqvae_checkpoint", default="/mnt/data/wildfire/runs/VQVAE/version_9/checkpoints/step=21500-fid_score=6.419.ckpt")
    parser.add_argument("--enc_folder", default="/mnt/data/wildfire/imgs1_enc")
    parser.add_argument("--dec_folder", default="/mnt/data/wildfire/imgs1_dec")
    parser.add_argument("--image_size", type=int, default=512)
    # imgs1
    parser.add_argument("--mean", type=float, default=2.7935)
    parser.add_argument("--std", type=float, default=5.9023)
    # imgs1_spot_subset
    # parser.add_argument("--mean", type=float, default=6.1121)
    # parser.add_argument("--std", type=float, default=5.2281)
    config = parser.parse_args()

    set_torch_options()

    enc_folder = Path(config.enc_folder)
    enc_folder.mkdir(parents=True, exist_ok=True)
    dec_folder = Path(config.dec_folder)
    dec_folder.mkdir(parents=True, exist_ok=True)

    ds = TemperatureDataWithIndices(config.in_folder)
    dl = DataLoader(
        ds,
        batch_size=16,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    vqvae: VQModel = load_ae_from_checkpoint(config.vqvae_checkpoint)
    vqvae.eval().cuda()

    stride = 2**(len(vqvae.encoder.down_blocks) - 1)
    latent_dim: int = vqvae.config.vq_embed_dim
    sample_size = int(config.image_size / stride)

    """
    VQVAE: stride S=8, latent dimension D=12
    Thermal images: HxW, float32  => 32*H*W bit
    Latent space: (H/S)x(W/S)xD, float32 => 32*D*H*W/S^2 bit
    Compression: D/S^2 => 18.75%
    Thus compressed latents take up only 18.75% of original disk space.
    
    With VQVAE setting of S=16, D=12 this would be then only
    4.68755% of original disk space.

    DJI_20231017140139_0126_T
    """
    print(f"Compression: {latent_dim / (stride ** 2) * 100}%")
    
    for x, idx in tqdm(dl, desc="Processing dataloader..."):  # BxHxW
        x = x.cuda()
        latents = encode(vqvae, x[:, None, :, :], config.mean, config.std)  # BxCxHxW
        for y, i in zip(latents, idx):
            p: Path = ds.files[int(i)]
            new_p = (enc_folder / p.stem).with_suffix(".npy")
            y = y.cpu().numpy()
            np.save(new_p, y)

        x = decode(vqvae, latents, config.mean, config.std)
        for y, i in zip(x, idx):
            p: Path = ds.files[int(i)]
            new_p = (dec_folder / p.stem).with_suffix(".png")
            y = y.cpu()
            image = tone_mapping(y, y.min(), y.max())
            image.save(new_p)
