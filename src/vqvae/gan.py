from torch import Tensor
import torch.nn.functional as F
import torch
from torch.autograd import grad
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class AdvConfig:
    disc_weight: float = 0.1
    start_step: int = 1500
    adaptive_weight: bool = False
    loss_type: str = "hinge"
    lecam_reg_weight: float = 1e-3
    ema_decay: float = 0.99
    n_layers: int = 2


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('GroupNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

    https://arxiv.org/pdf/2409.16211v1 MaskBit -> Discriminator Update.
    -> replace the 4x4 convolution kernels with 3x3 kernels
    -> switch to group normalization

    Currently the stride is 8, in MaskBit they proposed to add 2x2 max pooling
    to have same stride as "generator" which is the VQVAE encoder, but here I use
    a stride of 8 for the encoder anyway, so it is already symmetrical.
    -> 2x2 max pooling to align the output stride between the generator and discriminator

    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
        """
        super().__init__()
        self.stride = 8
        kw = 3
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.GroupNorm(ndf // 2, ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.GroupNorm(ndf // 2, ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        self.apply(weights_init)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


def generator_loss(logits: torch.Tensor, loss_type: str = "hinge"):
    """
    :param logits: discriminator output in the generator phase (fake_logits)
    :param loss_type: which loss to apply between "hinge" and "non-saturating"
    """
    if loss_type == "hinge":
        loss = -torch.mean(logits)
    elif loss_type == "non-saturating":
        loss = F.binary_cross_entropy_with_logits(logits, target=torch.ones_like(logits))
    else:
        raise ValueError(f"unknown loss_type: {loss_type}")
    return loss


def discriminator_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor, loss_type: str = "hinge"):
    """
    :param logits_real: discriminator output when input is the original image
    :param logits_fake: discriminator output when input is the reconstructed image
    :param loss_type: which loss to apply between "hinge" and "non-saturating"
    """

    if loss_type == "hinge":
        real_loss = F.relu(1.0 - logits_real)
        fake_loss = F.relu(1.0 + logits_fake)
    elif loss_type == "non-saturating":
        real_loss = F.binary_cross_entropy_with_logits(
            logits_real, target=torch.ones_like(logits_real), reduction="none"
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            logits_fake, target=torch.zeros_like(logits_fake), reduction="none"
        )
    else:
        raise ValueError(f"unknown loss_type: {loss_type}")

    return 0.5 * torch.mean(real_loss + fake_loss)


def lecam_reg_loss(real_pred, fake_pred, ema_real_pred, ema_fake_pred):
    """Lecam loss for data-efficient and stable GAN training.

    Described in 

    Args:
    real_pred: Prediction (scalar) for the real samples.
    fake_pred: Prediction for the fake samples.
    ema_real_pred: EMA prediction (scalar)  for the real samples.
    ema_fake_pred: EMA prediction for the fake samples.

    Returns:
    Lecam regularization loss (scalar).

    https://arxiv.org/pdf/2409.16211v1 MaskBit -> Discriminator Update.
    -> Additionally, we incorporate LeCAM loss to stabilize adversarial training. 
    (LeCAM: https://arxiv.org/abs/2104.03310)
    """
    lecam_loss = torch.mean(torch.pow(F.relu(real_pred - ema_fake_pred), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_real_pred - fake_pred), 2))
    return lecam_loss


class AdversarialLoss(nn.Module):

    def __init__(
        self,
        in_channels: int,
        disc_weight: float,
        start_step: int,
        adaptive_weight: bool,
        loss_type: str,
        lecam_reg_weight: float,
        ema_decay: float,
        n_layers: int,
    ):
        """
        Changes inspired by the Maskbit paper --> https://arxiv.org/pdf/2409.16211
        https://github.com/SerezD/vqvae-vqgan-pytorch-lightning/blob/master/vqvae/model.py
        """
        super().__init__()
        # NOTE: stride := 2^(n_layers)
        self.disc = NLayerDiscriminator(input_nc=in_channels, n_layers=n_layers)
        self.disc_weight = disc_weight
        self.start_step = start_step
        self.adaptive_weight = adaptive_weight
        self.loss_type = loss_type
        self.lecam_reg_weight = lecam_reg_weight
        self.ema_decay = ema_decay
        self.ema_logits_real = None
        self.ema_logits_fake = None
        self.last_logits_real = torch.zeros([1])
        self.last_logits_fake = torch.zeros([1])

    def calculate_adaptive_weight(self, nll_loss: float, g_loss: float, last_layer: torch.nn.Parameter):
        """
        From Taming Transformers for High-Resolution Image Synthesis paper, Patrick Esser, Robin Rombach, Bjorn Ommer:

        "we compute the adaptive weight λ according to λ = ∇GL[Lrec] / (∇GL[LGAN] + δ)
         where Lrec is the perceptual reconstruction loss, ∇GL[·] denotes the gradient of its input w.r.t. the last
         layer L of the decoder, and δ = 10−6 is used for numerical stability"

        https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
        """
        nll_grads = grad(nll_loss, last_layer, grad_outputs=torch.ones_like(nll_loss), retain_graph=True)[0].detach()
        g_grads = grad(g_loss, last_layer, grad_outputs=torch.ones_like(g_loss), retain_graph=True)[0].detach()

        adaptive_weight = torch.norm(nll_grads, p=2) / (torch.norm(g_grads, p=2) + 1e-8)
        adaptive_weight = torch.clamp(adaptive_weight, 0.0, 1e4).detach()
        adaptive_weight = adaptive_weight * self.disc_weight

        return adaptive_weight

    def forward_autoencoder(
        self,
        p_loss: float,  # perceptual loss in original implementation
        reconstructions: torch.Tensor,
        current_step: int,
        last_decoder_layer: torch.nn.Parameter,
    ):
        if current_step >= self.start_step:
            logits_fake = self.disc(reconstructions.contiguous())
            g_loss = generator_loss(logits_fake, loss_type=self.loss_type)

            if self.training and self.adaptive_weight:
                g_weight = self.calculate_adaptive_weight(
                    p_loss, g_loss, last_layer=last_decoder_layer
                )
            else:
                g_weight = self.disc_weight

        else:
            g_loss = torch.zeros_like(p_loss, requires_grad=False)
            g_weight = 0.0  # disc not started yet

        return g_loss, g_weight
    
    def update_ema_inplace(self, ema_tensor: Tensor, new_tensor: Tensor):
        ema_tensor.mul_(self.ema_decay)
        ema_tensor.add_((1 - self.ema_decay) * new_tensor)
    
    def calcualte_lecam_regularization_term(
            self, logits_real: torch.Tensor, logits_fake: torch.Tensor
        ):
        """
        https://arxiv.org/pdf/2104.03310
        https://github.com/google/lecam-gan/blob/master/stylegan2/lecam_loss.py
        """
        # In: https://github.com/google/lecam-gan/blob/master/stylegan2/lecam_loss.py
        # the mean of the logits (discriminator predictions) are tracked.
        logits_real = logits_real.detach().mean()
        logits_fake = logits_fake.detach().mean()

        self.last_logits_real = logits_real
        self.last_logits_fake = logits_fake

        if self.ema_logits_real is None:
            # Initialization for EMA tracking on first time.
            self.ema_logits_real = logits_real
            self.ema_logits_fake = logits_fake
            lecam_term = torch.zeros((1,), device=logits_real.device)
        else:
            # EMA update.
            self.update_ema_inplace(self.ema_logits_real, logits_real)
            self.update_ema_inplace(self.ema_logits_fake, logits_fake)
            lecam_term = lecam_reg_loss(
                logits_real, logits_fake, self.ema_logits_real, self.ema_logits_fake
            )

        return lecam_term

    def forward_discriminator(
        self,
        images: torch.Tensor,
        reconstructions: torch.Tensor,
        current_step: int,
        use_lecam_regularization: bool,
    ):
        if current_step >= self.start_step:
            logits_real = self.disc(images)
            logits_fake = self.disc(reconstructions.contiguous().detach())
            d_loss = discriminator_loss(logits_real, logits_fake, loss_type=self.loss_type)
            lecam_term = self.calcualte_lecam_regularization_term(logits_real, logits_fake)
            if use_lecam_regularization:
                loss = d_loss + self.lecam_reg_weight * lecam_term
            else:
                lecam_term = torch.zeros((1,), device=images.device)
                loss = d_loss

        else:
            device = images.device
            d_loss = torch.zeros((1,), device=device)
            lecam_term = torch.zeros((1,), device=device)
            loss = None

        return loss, d_loss, lecam_term
