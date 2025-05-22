import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F

"""
https://huggingface.co/docs/diffusers/main/api/normalization

Diffusion Transformer (DiT) paper introduces adaLN-Zero
https://arxiv.org/pdf/2212.09748
"""

# from diffusers.models.normalization


ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
    

class AdaGroupNorm(nn.Module):
    r"""
    GroupNorm layer modified to condition on embeddings.

    Parameters:
        embed_dim (`int`): The size of each embedding vector.
        in_channels (`int`): The size of each input vector.
        num_groups (`int`): The number of groups to separate the channels into.
        act_fn (`str`, *optional*, defaults to `None`): The activation function to use.
        eps (`float`, *optional*, defaults to `1e-5`): The epsilon value to use for numerical stability.
    """

    def __init__(
        self, embed_dim: int, in_channels: int, num_groups: int, act_fn: Optional[str] = None, eps: float = 1e-5
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.in_channels = in_channels

        if act_fn is None:
            self.act = None
        else:
            self.act = get_activation(act_fn)

        self.linear = nn.Linear(embed_dim, in_channels * 2)
        
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)  # BxCx1x1

        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        return x
    

class AdaLayerNorm(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    IMPORTANT: This custom implementation expects input to be BxHxWxC to work.

    Parameters:
        in_channels (`int`): The num. channels of the input image x.
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
        zero (`bool`): If False adaLN, if True adaLN-Zero.

    The adaLN-Zero computes an additional scale parameter. The DiT block
    first applies layer normalization, scales and shifts this output and then
    feeds it to MHSA/FFN and at the end scales the output of MHSA/FFN with the
    other scale it computed from the condition. Thus, if zero=True the linear layer
    predicts a vector with 3x in_channels dimension (otherwise 2x in_channels). 

    If num_embeddings is given we learn an embedding table for each normalization layer.

    If embedding_dim is not given, we only apply a regular LayerNorm (non-adaptive).
    """

    def __init__(
        self,
        in_channels: int,
        embedding_dim: Optional[int] = None,
        num_embeddings: Optional[int] = None,
        bias: bool = True,  # linear layer bias for embedding transformation
        zero: bool = False,  # DiT has adaLN-Zero
    ):
        super().__init__()
        if embedding_dim is None:
            self.norm = nn.LayerNorm(in_channels, elementwise_affine=True, eps=1e-6)
        else:  # adaptive
            if num_embeddings is not None:
                self.emb = nn.Embedding(num_embeddings, embedding_dim)
            else:
                self.emb = None

            self.silu = nn.SiLU()
            self.multiplier = 3 if zero else 2
            self.linear = nn.Linear(embedding_dim, self.multiplier * in_channels, bias=bias)
            self.norm = nn.LayerNorm(in_channels, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        class_labels: Optional[torch.LongTensor] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE: BxHxWxC
        if self.emb is not None:  # AdaLayerNorm with individual embeddings.
            emb = self.emb(class_labels)

        if emb is not None:  # AdaLayerNorm with shared embeddings.
            emb = self.linear(self.silu(emb))
            if self.multiplier == 2:
                shift, scale = emb.chunk(self.multiplier, dim=1)  # BxC
                gate = 1.0
            else:
                shift, scale, gate = emb.chunk(self.multiplier, dim=1)  # BxC
                gate = gate[:, None, None, :]  # Bx1x1xC
            x = self.norm(x) * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
            return x, gate
        else:  # Just a regular LayerNorm.
            x = self.norm(x)
            return x, 1.0


class SpatialNorm(nn.Module):
    """
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        f_size = f.shape[-2:]
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f
    
