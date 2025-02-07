import argparse
from pathlib import Path
import torch
from torch import Tensor
from tqdm import tqdm
from PIL import Image
from typing import Union
import os
import torch.distributed as dist

from ..ldm.train import (
    LDM,
    load_ldm_from_checkpoint,
    TrainConfig,
    AutoEncoderType,
)
from ..utils.torch_conf import set_torch_options


def get_cpu_count():
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    return cpu_count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", help="Where to save the generated images?")
    parser.add_argument("ldm_checkpoint", help="Path to a *.ckpt file")
    parser.add_argument("num_images", type=int, help="How many images should be generated?")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_type", default="raw", help="In which format to save new data?")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    return args


def setup_model_for_inference(ldm_checkpoint: str, trace: bool, rank: int = 0) -> LDM:
    set_torch_options()
    
    if rank != 0:
        map_location = {"cuda:0": f"cuda:{rank}"}
    else:
        map_location = None

    model: LDM = load_ldm_from_checkpoint(
        ldm_checkpoint, map_location=map_location
    ).eval()
    model.requires_grad_(False)

    if trace:  # Optimize inference?
        latent_dim = model.vqvae.config.vq_embed_dim
        device = next(iter(model.parameters())).device
        latent_size = model.unet.sample_size

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

    return model


def run_inference(
    model: LDM,
    save_dir: str,
    num_images: int,
    batch_size: int,
    seed: int,
    output_type: str,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    device = next(model.parameters()).device

    n_images = 0
    pbar = tqdm(desc=f"Generate images ({device})...", total=num_images)
    while n_images < num_images:
        batch_size = min(num_images - n_images, batch_size)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            images: Union[Image.Image, Tensor] = model.generate_new_images(
                batch_size, generator, output_type, show_progress=False
            )
    
        if output_type == "pil":
            ext = "png"
        elif output_type == "raw":
            # Bx1xHxW to BxHxW
            images = images.to(device="cpu", dtype=torch.float32).numpy()  # Bx1xHxW
            images = [Image.fromarray(img[0, :, :]) for img in images]
            ext = "TIF"
        else:
            raise ValueError(f"Unknown: {output_type=}")
        
        for idx, image in enumerate(images):
            image.save(save_dir / f"{n_images + idx:07d}.{ext}")
        
        pbar.update(n=batch_size)
        n_images += batch_size


if __name__ == "__main__":
    """
    Inference on 1 node (=1 GPU cluster) with --nproc-per-node 4
    we would be using 4 GPUs within this cluster.
    The GPUs on the node get each a fraction of the overall available
    CPUs to work with by setting the environment variable OMP_NUM_THREADS
    properly. LOCAL refers to "within a cluster", while GLOBAL means
    "overall clusters".

    Example using only 1 GPU:
    torchrun --nproc-per-node 1 -m src.scripts.generate ...
    """
    args = parse_args()

    cpu_count = get_cpu_count()
    world_size = int(os.environ["LOCAL_WORLD_SIZE"])  # Equals: --nproc-per-node

    # NOTE: GLOBAL rank != LOCAL rank if multiple nodes would be used.
    rank = int(os.environ["RANK"]) 
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", device_id=device)

    if rank == 0:
        cpu_per_gpu = int(cpu_count / world_size)  # Recommandation!
        print(
            f"Recommended to set OMP_NUM_THREADS={cpu_per_gpu}"
            f" when using --nproc-per-node {world_size} as specified.\n"
            "Example:\n"
            f"OMP_NUM_THREADS={cpu_per_gpu} torchrun --nproc-per-node {world_size} -m src.scripts.generate ..."
        )

    print(f"{cpu_count=} {world_size=} {rank=}")

    model = setup_model_for_inference(args.ldm_checkpoint, args.trace, rank=rank)

    run_inference(
        model,
        args.save_dir,
        args.num_images,
        args.batch_size,
        args.seed + 46873 * rank,
        args.output_type,
    )

    dist.destroy_process_group()
