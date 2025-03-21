import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import os
from omegaconf import OmegaConf
from dataclasses import dataclass, field
import numpy as np
import random
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from .ae import Autoencoder
from ..utils.image import tone_mapping, pil_make_grid


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
    channels: int = 128
    residual_blocks: int = 2
    residual_channels: int = 32
    embed_dim: int = 64


@dataclass
class TrainConf:
    seed: int = 42
    device: str = "cuda"
    train_batch_size: int = 16
    eval_batch_size: int = 128
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.99
    gradient_clip_val: float = 1.0
    warmup_steps: int = 100
    train_steps: int = 4900
    num_workers: int = 0
    log_every_n_steps: int = 100
    val_every_n_steps: int = 500
    model: ModelConf = field(default_factory=ModelConf)


class Trainer:

    def __init__(self, conf: TrainConf):
        self.model = Autoencoder(in_channel=1, **conf.model)
        self.conf = conf
        

if __name__ == "__main__":
    """
    python -m src.ir.train
    """
    conf = OmegaConf.merge(
        OmegaConf.structured(TrainConf()),
        OmegaConf.load("src/configs/ir/ae.yaml"),
    )
    setup_torch(conf.seed)
    

# # Initialize Accelerator and TensorBoard writer
# accelerator = Accelerator()
# writer = SummaryWriter("runs/autoencoder")

# # Load dataset and apply transformations
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 images
# ])

# dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
# val_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# # Initialize model, loss function, and optimizer
# model = Autoencoder()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Prepare everything with Accelerator
# model, optimizer, dataloader, val_dataloader = accelerator.prepare(model, optimizer, dataloader, val_dataloader)

# # Create checkpoint directory
# os.makedirs("checkpoints", exist_ok=True)

# # Training loop
# def train_model(epochs=10):
#     best_val_loss = float('inf')
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for data, _ in dataloader:
#             optimizer.zero_grad()
#             outputs = model(data)
#             loss = criterion(outputs, data)
#             accelerator.backward(loss)
#             optimizer.step()
#             total_loss += loss.item()
#         avg_train_loss = total_loss / len(dataloader)
#         writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        
#         # Validation step
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for val_data, _ in val_dataloader:
#                 val_outputs = model(val_data)
#                 val_loss += criterion(val_outputs, val_data).item()
#         avg_val_loss = val_loss / len(val_dataloader)
#         writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        
#         # Save best model
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(model.state_dict(), "checkpoints/best_autoencoder.pth")
        
#         # Log images
#         images = val_data.view(-1, 1, 28, 28)[:8]  # Take first 8 images
#         reconstructions = val_outputs.view(-1, 1, 28, 28)[:8]
#         comparison = torch.cat([images, reconstructions], dim=0)  # Stack input and output images
#         grid = vutils.make_grid(comparison, nrow=8, normalize=True)
#         writer.add_image("Reconstruction", grid, epoch)
        
#         print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
#     writer.close()

# # Run training
# train_model(10)