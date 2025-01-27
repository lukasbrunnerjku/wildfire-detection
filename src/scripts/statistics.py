import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.image import TemperatureData


def main(folder: str):
    ds = TemperatureData(folder)
    dl = DataLoader(ds, batch_size=16, num_workers=4, drop_last=False)

    psum    = torch.tensor([0.0])
    psum_sq = torch.tensor([0.0])
    min_value = torch.tensor(torch.inf)
    max_value = torch.tensor(-torch.inf)

    for inputs in tqdm(dl, desc="Processing batches..."):
        inputs: Tensor

        if inputs.ndim == 3:  # BxHxW; single channel image
            inputs = inputs[:, None, :, :]  # Bx1xHxW

        psum    += inputs.sum(dim        = [0, 2, 3])
        psum_sq += (inputs ** 2).sum(dim = [0, 2, 3])

        min_value = torch.minimum(torch.min(inputs.view(-1)), min_value)
        max_value = torch.maximum(torch.max(inputs.view(-1)), max_value)

    H, W = inputs.shape[-2:]
    n_pixel = len(ds) * H * W

    # mean := E[X]
    # std := sqrt(E[X^2] - (E[X])^2)
    total_mean = psum / n_pixel
    total_var  = (psum_sq / n_pixel) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    print("### Statistics ###")
    print("- mean: {:.4f}".format(total_mean.item()))
    print("- std:  {:.4f}".format(total_std.item()))
    print("- min:  {:.4f}".format(min_value.item()))
    print("- max:  {:.4f}".format(max_value.item()))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()

    main(args.folder)