import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PartialResnet(nn.Module):

    def __init__(self, full_resnet: models.ResNet, up_to_layer: int):
        super().__init__()
        assert up_to_layer in (0, 1, 2, 3, 4)
        self.up_to_layer = up_to_layer

        if self.up_to_layer >= 0:
            self.layer0 = nn.Sequential(
                full_resnet.conv1,
                full_resnet.bn1,
                full_resnet.relu,
                full_resnet.maxpool,
            )
        if self.up_to_layer >= 1:
            self.layer1 = full_resnet.layer1
        if self.up_to_layer >= 2:
            self.layer2 = full_resnet.layer2
        if self.up_to_layer >= 3:
            self.layer3 = full_resnet.layer3
        if self.up_to_layer >= 4:
            self.layer4 = full_resnet.layer4

    def forward(self, x):
        if self.up_to_layer >= 0:
            x = self.layer0(x)
        if self.up_to_layer >= 1:
            x = self.layer1(x)
        if self.up_to_layer >= 2:
            x = self.layer2(x)
        if self.up_to_layer >= 3:
            x = self.layer3(x)
        if self.up_to_layer >= 4:
            x = self.layer4(x)
        return x
    

class PerceptualLoss(nn.Module):

    def __init__(self, resnet_type: str = "resnet50", up_to_layer: Optional[int] = None):
        super().__init__()
        # https://arxiv.org/pdf/2409.16211v1 MaskBit -> Perceptual Loss.
        # -> L2 loss on logits of pretrained ResNet50 of original and
        # reconstructed images
        if resnet_type == "resnet50":
            full_resnet = models.resnet50(models.ResNet50_Weights.DEFAULT)
        elif resnet_type == "resnet18":
            full_resnet = models.resnet18(models.ResNet18_Weights.DEFAULT)
        elif resnet_type == "resnet34":
            full_resnet = models.resnet34(models.ResNet34_Weights.DEFAULT)
        else:
            raise ValueError(f"Cannot handle specified resnet_type: {resnet_type}")
        
        if up_to_layer is not None:
            self.resnet = PartialResnet(full_resnet, up_to_layer)
        else:
            self.resnet = full_resnet
        
        # Freeze model parameters.
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )  # Expects BxCxHxW; [0., 1.] input.
        
    def train(self, mode: bool = True):
        mode = False  # Keep in eval mode, always.
        return super().train(mode)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, do_normalize: bool = False
    ) -> torch.Tensor:
        # Grayscale images? --> Expand channels.
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)  # BxCxHxW
        if y.shape[1] == 1:
            y = y.expand(-1, 3, -1, -1)  # BxCxHxW

        if do_normalize:  # Temperatures will be normalized differently.
            x = self.normalize(x)
            y = self.normalize(y)

        x_logits = self.resnet(x)
        y_logits = self.resnet(y)

        loss = F.mse_loss(x_logits, y_logits, reduction="mean")

        return loss
