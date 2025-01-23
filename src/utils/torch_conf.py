import torch
import numpy as np
import random
import os


def manual_seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_torch_options():
    """
    Speed up trainig/inference, optimize memory usage.
    NOTE: Tested only with NVIDIA A100 GPU!
    """
    if torch.cuda.is_available():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
