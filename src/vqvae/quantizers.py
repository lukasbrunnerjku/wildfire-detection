import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from vector_quantize_pytorch import FSQ, LFQ
from enum import Enum


def norm_square(x, y):
    """x: (N, D) / y: (M, D)"""
    N, D = x.shape
    M = y.shape[0]
    return F.mse_loss(
        x[:, None].expand(N, M, D), y[None, :].expand(N, M, D),
        reduction="none",
    ).sum(dim=-1)


def identity(x: Tensor, out: Tensor=None) -> Tensor:
    if out is not None:
        out.copy_(x)
        return out
    return x


class NSVQ(nn.Module):
    """
    NSVQ: Noise Substitution in Vector Quantization [2022 paper]
    https://ieeexplore.ieee.org/document/9696322  
    """
    def __init__(self, codebook_size: int, codebook_dim: int, normalize=None):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        self.normalize = identity if normalize is None else normalize

        nn.init.normal_(self.embedding.weight)
        with torch.no_grad():
            self.normalize(self.embedding.weight, out=self.embedding.weight)

    @property
    def code(self):
        return self.embedding.weight

    def __len__(self):
        return self.code.shape[0]

    def forward(self, z: Tensor):
        B, D, H, W = z.shape
        z = z.permute(0, 2, 3, 1).contiguous()  # BxHxWxD
        z_flattened = z.view(-1, self.codebook_dim)  # (N, D)

        with torch.no_grad():
            self.normalize(self.code, out=self.code)
        dist = norm_square(z_flattened, self.code)
        # (N, D) x (size, D) -> (N, size) -> (N, 1)
        dist_min, ind = torch.min(dist, dim=1, keepdim=True)

        # Compatibility with VectorQuantizer from Huggingface!
        ind = ind[:, 0]  # N,

        if self.training:
            random_vector = torch.empty_like(z_flattened).normal_()
            norm_random_vector = torch.norm(random_vector, dim=-1, keepdim=True)
            norm_random_vector = torch.maximum(norm_random_vector, torch.full_like(norm_random_vector, 1e-9))

            q_error = (random_vector / norm_random_vector) * dist_min.sqrt()
            z_q = z_flattened + q_error  # (N, D)

            z_q_view = z_q.view(B, D, H * W)
            loss = (z_q_view.permute(0, 2, 1) @ z_q_view).mean()

            # Reshape back to match original input shape
            z_q = z_q.view(B, H, W, D).permute(0, 3, 1, 2)  # BxDxHxW

            return z_q, loss, ind
        
        else:
            z_q = self.embedding(ind)
            # Reshape back to match original input shape
            z_q = z_q.view(B, H, W, D).permute(0, 3, 1, 2)  # BxDxHxW
            return z_q, None, ind
        


class CodebookSize(Enum):
    SMALL = 256
    MEDIUM = 1024
    LARGE = 4096
    HUGE = 16384


class QuantizerType(Enum):
    NSVQ = 0
    LFQ = 1
    FSQ = 2


class Quantizer(nn.Module):

    def __init__(self, quantizer_type: QuantizerType, base_two_exponent: int):
        """
        The forward method is compatible with the hugginface diffusers
        VQModel .quantize quantizer and can be seamlessly swapped with any
        of the implemented quantizer types.

        base_two_exponent:
            8 ... 2^8 --> 256 codebook size
            10 ... 2^10 --> 1024 codebook size
            12 ... 2^12 --> 4096 codebook size
            14 ... 2^14 --> 16384 codebook size
        """
        super().__init__()
        assert base_two_exponent in (8, 10, 12, 14)
        self.quantizer_type = quantizer_type

        if quantizer_type == QuantizerType.FSQ:
            # https://arxiv.org/pdf/2309.15505
            if base_two_exponent == 8:
                levels = [8, 6, 5]
            elif base_two_exponent == 10:
                levels = [8, 5, 5, 5]
            elif base_two_exponent == 12:
                levels = [7, 5, 5, 5, 5]
            else:
                levels = [8, 8, 8, 6, 5]
            self.quantizer = FSQ(levels)
        elif quantizer_type == QuantizerType.LFQ:
            self.quantizer = LFQ(dim=base_two_exponent, spherical=False)
        elif quantizer_type == QuantizerType.NSVQ:
            self.quantizer = NSVQ(
                codebook_size=int(2**base_two_exponent),
                codebook_dim=3,
                normalize=F.normalize,
            )
        else:
            raise ValueError(f"Cannot handle quantizer_type: {quantizer_type}")
        
        print(f"{self.quantizer.codebook_size=} {self.quantizer.codebook_dim=}")

    def forward(self, z: Tensor):  # BxCxHxW
        perplexity, min_encodings = None, None

        if self.quantizer_type == QuantizerType.FSQ:
            z_q, ind = self.quantizer(z)
            ind = ind.view(-1)  # BxHxW --> N,
            loss = torch.zeros([1], device=z_q.device)
        elif self.quantizer_type == QuantizerType.LFQ:
            ret = self.quantizer(z)
            z_q = ret.quantized
            ind = ret.indices
            ind = ind.view(-1)  # BxHxW --> N,
            loss = ret.entropy_aux_loss
        elif self.quantizer_type == QuantizerType.NSVQ:
            z_q, loss, ind = self.quantizer(z)

        # assert z_q.shape == z.shape  # BxCxHxW
        # assert ind.ndim == 1  # N,

        # Provide compatible return values with VQModel.quantize model.
        return z_q, loss, (perplexity, min_encodings, ind)


def perplexity(hist: Tensor):
    hist = hist[hist > 0.0]
    hist = hist / hist.sum()  # normalize
    return (-hist * torch.log(hist)).sum().exp()


def add_histogram_raw(writer, tag: str, hist: Tensor, global_step: int):
    num_bins = len(hist)
    num = hist.sum()
    sum_values = torch.dot(torch.arange(num_bins), hist)
    sum_squares = torch.dot(torch.arange(num_bins) ** 2, hist)
    bucket_limits = torch.linspace(0.5, num_bins - 0.5, num_bins)
    writer.add_histogram_raw(
        tag,
        min=0,
        max=num_bins - 1,
        num=num,
        sum=sum_values,
        sum_squares=sum_squares,
        bucket_limits=bucket_limits.tolist(),
        bucket_counts=hist.tolist(),
        global_step=global_step,
    )
