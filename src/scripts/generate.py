import argparse
from pathlib import Path
import torch
from torch import Tensor
from tqdm import tqdm
from PIL import Image
import gc
from typing import Union

from ..ldm.train import (
    LDM,
    load_ldm_from_checkpoint,
    TrainConfig,
    AutoEncoderType,
)
from ..utils.torch_conf import set_torch_options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", help="Where to save the generated images?")
    parser.add_argument("ldm_checkpoint", help="Path to a *.ckpt file")
    parser.add_argument("num_images", type=int, help="How many images should be generated?")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_type", default="raw", help="In which format to save new data?")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    set_torch_options()

    model: LDM = load_ldm_from_checkpoint(args.ldm_checkpoint).eval()
    model.requires_grad_(False)

    latent_dim = model.vqvae.config.vq_embed_dim
    device = next(iter(model.parameters())).device
    latent_size = model.unet.sample_size

    if args.trace:  # Optimize inference?
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            scripted_model = torch.jit.trace_module(
                model.vqvae,
                inputs={"decode": torch.randn(1, latent_dim, latent_size, latent_size, device=device)},
                strict=False,
            )
            scripted_model = torch.jit.optimize_for_inference(scripted_model, ["decode"])
            model.vqvae = scripted_model

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            scripted_model = torch.jit.trace_module(
                model.unet,
                inputs={
                    "forward": (
                        torch.randn(1, latent_dim, latent_size, latent_size, device=device),
                        torch.tensor([1.0], dtype=torch.long, device=device),
                    )
                },
                strict=False,
            )
            scripted_model = torch.jit.optimize_for_inference(scripted_model, ["forward"])
            model.unet = scripted_model
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    
    n_images = 0
    pbar = tqdm(desc="Generating images...", total=args.num_images)
    while n_images < args.num_images:
        batch_size = min(args.num_images - n_images, args.batch_size)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            images: Union[Image.Image, Tensor] = model.generate_new_images(
                batch_size, generator, args.output_type
            )
    
        if args.output_type == "pil":
            ext = "png"
        elif args.output_type == "raw":
            # Bx1xHxW to BxHxW
            images = images.to(device="cpu", dtype=torch.float32).numpy()  # Bx1xHxW
            images = [Image.fromarray(img[0, :, :]) for img in images]
            ext = "TIF"
        else:
            raise ValueError(f"Unknown: {args.output_type=}")
        
        for idx, image in enumerate(images):
            image.save(save_dir / f"{n_images + idx:07d}.{ext}")

        # gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        
        pbar.update(n=batch_size)
        n_images += batch_size
