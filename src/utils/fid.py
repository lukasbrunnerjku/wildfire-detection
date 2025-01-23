from copy import deepcopy
from torchmetrics.image.fid import (
    _compute_fid,
    _FeatureExtractorInceptionV3,
    interpolate_bilinear_2d_like_tensorflow1x,
)
from torchmetrics.image.fid import NoTrainInceptionV3, Metric
from typing import List, Optional, Tuple, Union, Any
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class _NoTrainInceptionV3(_FeatureExtractorInceptionV3):
    """Module that never leaves evaluation mode."""

    def __init__(
        self,
        name: str,
        features_list: List[str],
        feature_extractor_weights_path: Optional[str] = None,
    ) -> None:
        super().__init__(name, features_list, feature_extractor_weights_path)
        # put into evaluation mode
        self.eval()

    def train(self, mode: bool) -> "_NoTrainInceptionV3":
        """Force network to always be in evaluation mode."""
        return super().train(False)

    def _torch_fidelity_forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward method of inception net.

        Copy of the forward method from this file:
        https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/feature_extractor_inceptionv3.py
        with a single line change regarding the casting of `x` in the beginning.

        Corresponding license file (Apache License, Version 2.0):
        https://github.com/toshas/torch-fidelity/blob/master/LICENSE.md

        """
        features = {}
        remaining_features = self.features_list.copy()

        x = x.to(self._dtype) if hasattr(self, "_dtype") else x.to(torch.float)
        x = interpolate_bilinear_2d_like_tensorflow1x(
            x,
            size=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE),
            align_corners=False,
        )

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.MaxPool_1(x)

        if "64" in remaining_features:
            features["64"] = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
            remaining_features.remove("64")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.MaxPool_2(x)

        if "192" in remaining_features:
            features["192"] = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
            remaining_features.remove("192")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        if "768" in remaining_features:
            features["768"] = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
            remaining_features.remove("768")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.AvgPool(x)
        x = torch.flatten(x, 1)

        if "2048" in remaining_features:
            features["2048"] = x
            remaining_features.remove("2048")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        if "logits_unbiased" in remaining_features:
            x = x.mm(self.fc.weight.T)
            # N x 1008 (num_classes)
            features["logits_unbiased"] = x
            remaining_features.remove("logits_unbiased")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

            x = x + self.fc.bias.unsqueeze(0)
        else:
            x = self.fc(x)

        features["logits"] = x
        return tuple(features[a] for a in self.features_list)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of neural network with reshaping of output."""
        out = self._torch_fidelity_forward(x)
        return out[0].reshape(x.shape[0], -1)
    

class FID(Metric):

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    inception: nn.Module
    feature_network: str = "inception"

    def __init__(
        self,
        feature: Union[int, nn.Module] = 2048,
        reset_real_features: bool = True,
        input_img_size: Tuple[int, int, int] = (1, 299, 299),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.used_custom_model = False

        if isinstance(feature, int):
            num_features = feature
            valid_int_input = (64, 192, 768, 2048)
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}."
                )

            self.inception = _NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(feature)])

        elif isinstance(feature, nn.Module):
            self.inception = feature
            self.used_custom_model = True
            if hasattr(self.inception, "num_features"):
                num_features = self.inception.num_features
            else:
                if self.normalize:
                    dummy_image = torch.rand(1, *input_img_size, dtype=torch.float32)
                else:
                    dummy_image = torch.randint(0, 255, (1, *input_img_size), dtype=torch.uint8)
                num_features = self.inception(dummy_image).shape[-1]
        else:
            raise TypeError("Got unknown input to argument `feature`")
        
        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        mx_num_feats = (num_features, num_features)
        self.add_state("real_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("real_features_cov_sum", torch.zeros(mx_num_feats).double(), dist_reduce_fx="sum")
        self.add_state("real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.add_state("fake_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_cov_sum", torch.zeros(mx_num_feats).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

    def update(self, imgs: Tensor, real: bool) -> None:
        """Update the state with extracted features.

        Args:
            imgs: Input img tensors to evaluate. If used custom feature extractor please
                make sure dtype and size is correct for the model.
            real: Whether given image is real or fake.

        """
        features = self.inception(imgs)
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += imgs.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += imgs.shape[0]

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(self.orig_dtype)

    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_cov_sum = deepcopy(self.real_features_cov_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
        else:
            super().reset()

    def set_dtype(self, dst_type: Union[str, torch.dtype]) -> "Metric":
        """Transfer all metric state to specific dtype. Special version of standard `type` method.

        Arguments:
            dst_type: the desired type as ``torch.dtype`` or string
        
        """
        out = super().set_dtype(dst_type)
        if isinstance(out.inception, NoTrainInceptionV3):
            out.inception._dtype = dst_type
        return out
    

def test_mock():
    x = torch.randn(128, 3, 512, 512)
    y = torch.randn(128, 3, 512, 512)

    fid = FID()
    fid.update(x, real=True)
    fid.update(y, real=False)
    fid_score = fid.compute()
    print(f"Mock: {fid_score=}")


@torch.no_grad()
def test_real(
    image_foder: str,
    image_mean: float,
    image_std: float,
    image_size: int = 512,
    num_samples: int = 512,
    batch_size: int = 128,
):
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from torchvision import transforms
    from torchmetrics.image.fid import FrechetInceptionDistance

    from .image import normalize_tif, TemperatureData, tone_mapping

    ds = TemperatureData(image_foder)
    dl = DataLoader(ds, batch_size=1, num_workers=32, drop_last=False, shuffle=True)

    crop_fn = transforms.CenterCrop(image_size)

    images = []
    for i, batch in enumerate(tqdm(dl, desc="Processing batches...")):
        images.append(crop_fn(batch))  # BxHxW
        if i >= (num_samples - 1):
            break

    # gFID
    x = torch.concat(
        images[0:num_samples:2], dim=0
    ).reshape(
        int(num_samples / 2), 1, image_size, image_size
    ).repeat(1, 3, 1, 1)  # Bx3xHxW

    y = torch.concat(
        images[1:num_samples:2], dim=0
    ).reshape(
        int(num_samples / 2), 1, image_size, image_size
    ).repeat(1, 3, 1, 1)  # Bx3xHxW

    # rFID
    # x = torch.concat(images, dim=0).reshape(-1, 1, image_size, image_size).repeat(
    #     1, 3, 1, 1
    # )[:int(num_samples / 2), :, :, :]  # Bx3xHxW
    # y = x.clone()

    fid = FrechetInceptionDistance().cuda()
    fid.set_dtype(torch.float32)
    fid.inception.INPUT_IMAGE_SIZE = image_size
    for i in range(0, len(x), batch_size):
        fid.update(tone_mapping(x[i:i+batch_size], return_tensor=True).cuda(), real=True)
        fid.update(tone_mapping(y[i:i+batch_size], return_tensor=True).cuda(), real=False)
    fid_score = fid.compute().cpu()
    print(f"With tone mapping: {fid_score=}")

    fid = FID().cuda()
    fid.set_dtype(torch.float32)
    fid.inception.INPUT_IMAGE_SIZE = image_size
    for i in range(0, len(x), batch_size):
        fid.update(normalize_tif(x[i:i+batch_size], image_mean, image_std).cuda(), real=True)
        fid.update(normalize_tif(y[i:i+batch_size], image_mean, image_std).cuda(), real=False)
    fid_score = fid.compute().cpu()
    print(f"With normalization: {fid_score=}")

    fid = FID().cuda()
    fid.set_dtype(torch.float32)
    fid.inception.INPUT_IMAGE_SIZE = image_size
    for i in range(0, len(x), batch_size):
        fid.update(x[i:i+batch_size].cuda(), real=True)
        fid.update(y[i:i+batch_size].cuda(), real=False)
    fid_score = fid.compute().cpu()
    print(f"Without normalization: {fid_score=}")


def build_fid_metric(
    image_size: int = 512,
    reset_real_features: bool = False,
    fid_type: torch.dtype = torch.float32,
) -> FID:
    """
    FID calculation depens further on the "batch size" used to
    "update" the real and fake images!
    Take a "batch size" >= 128 and unormalized temperature images.
    
    NOTE: Expects Bx3xHxW images so use x.repeat(1, 3, 1, 1) with x beeing
    of shape: Bx1xHxW
    """
    fid = FID(reset_real_features=reset_real_features)
    fid.set_dtype(fid_type)
    fid.inception.INPUT_IMAGE_SIZE = image_size
    return fid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", default="/mnt/data/wildfire/imgs1")
    parser.add_argument("--image_mean", type=float, default=2.7935)
    parser.add_argument("--image_std", type=float, default=5.9023)
    args = parser.parse_args()

    test_mock()
    test_real(args.image_folder, args.image_mean, args.image_std)
