from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torch import Tensor
import torch
import numpy as np
from lightning.pytorch import LightningDataModule
import torchvision.transforms.functional as TF


class RandomRotate90(torch.nn.Module):
    """Rotate the given image randomly +-90 degrees with given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            if torch.rand(1) < 0.5:
                angle = 90.0
            else:
                angle = -90.0
            return TF.rotate(img, angle)
        
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class CropDataset(Dataset):

    def __init__(
        self,
        folder: str,
        image_size: int,
        num_crops: int,
        deterministic: bool = False,
        do_resize: bool = True,
        extention: str = "png",
    ) -> None:
        super().__init__()
        self.num_crops = num_crops
        self.do_resize = do_resize
        self.paths = [p for p in Path(f"{folder}").glob(f"*.{extention}")]
        
        self.resize = transforms.Resize(image_size)

        if deterministic:  # evaluation
            self.crop_fn = transforms.CenterCrop(image_size)
        else:  # training
            self.crop_fn = transforms.Compose(
                [
                    transforms.RandomCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    RandomRotate90(),
                ]
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) -> list[list[Tensor]]:
        path = self.paths[index]
        img = Image.open(path)
        if self.do_resize:
            img: Image.Image = self.resize(img)  # HWC or HW
        img = torch.from_numpy(np.array(img, copy=True))
        if img.ndim == 2:
            img = img.unsqueeze(0)  # CHW
        else:
            img = img.permute(2, 0, 1)  # CHW
        crops = [self.crop_fn(img) for _ in range(self.num_crops)]
        return [crops,]  # N x R x CHW
    

def crop_collate(batch):  # B x N x R x CHW
    N = len(batch[0])
    all_chw = [[] for _ in range(N)]
    for nrchw in batch:
        for n in range(N):
            all_chw[n].extend(nrchw[n])  # R x CHW
        
    for n in range(N):
        all_chw[n] = torch.stack(all_chw[n], dim=0)  # (B*R)CHW
    
    N = len(all_chw)

    if N > 1:
        return all_chw  # N x (B*R)CHW
    else:
        return all_chw[0]  # (B*R)CHW


def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)


class DataModule(LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage: str) -> None:
        train_ds = CropDataset(
            self.config.data.folder,
            self.config.data.image_size,
            self.config.data.num_crops,
            do_resize=self.config.data.do_resize,
            extention=self.config.data.extention,
        )
        val_ds = CropDataset(
            self.config.data.folder,
            self.config.data.image_size,
            1,
            deterministic=True,
            do_resize=self.config.data.do_resize,
            extention=self.config.data.extention,
        )
        if len(val_ds) > self.config.data.num_fid_samples:
            indices = torch.randperm(len(val_ds))[:self.config.data.num_fid_samples]
            val_ds = Subset(val_ds, indices)
        
        self.train_ds = train_ds
        self.val_ds = val_ds

    def train_dataloader(self):
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.config.train_batch_size // self.config.data.num_crops,
            shuffle=True,
            pin_memory=True,
            num_workers=self.config.num_workers,
            collate_fn=crop_collate,
        )
        train_dl = InfiniteDataloader(train_dl)
        return train_dl
    
    def val_dataloader(self):
        val_dl = DataLoader(
            self.val_ds,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.config.num_workers,
            collate_fn=crop_collate,
        )
        return val_dl


if __name__ == "__main__":
    ds = CropDataset(
        folder="/mnt/data/wildfire/imgs1",
        image_size=128,
        num_crops=9,
        deterministic=False,
        do_resize=False,
        extention="TIF",
    )
    item = ds[0]  # N x R x CHW
    img = item[0][0]

    # ie. torch.Size([1, 128, 128]) torch.float32 tensor(-8.7000) tensor(14.4000)
    print(img.shape, img.dtype, img.min(), img.max())
