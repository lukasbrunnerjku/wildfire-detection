from torch.utils.data import Dataset
import json
from pathlib import Path
from typing import Optional
import random

from .utils import load_xy


class AOSDataset(Dataset):
    def __init__(
        self,
        folders: Optional[list[Path]] = None,
        meta_path: Optional[Path] = None,
        key: str = "score",
        threshold: float = 0.33,
        normalized: bool = False,
    ):
        self.normalized = normalized
        
        if folders is not None and meta_path is not None:
            raise ValueError("Either folders or meta_path must be provided, not both.")
        
        if folders is not None:
            self.folders = folders
        elif meta_path is not None:
            with open(meta_path, "r") as fp:
                meta_list = json.load(fp)

            n_unfiltered = len(meta_list)
            meta_list = list(filter(lambda x: x[key] > threshold, meta_list))
            self.folders = [Path(x["folder"]) for x in meta_list]
            print(f"Data Utilization: {100*len(self.folders)/n_unfiltered}%")
        else:
            raise ValueError("Either folders or meta_path must be provided.")

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]  # ie. /Batch-1/27/abc30f62e9ba47cab97c3ef3850a114d
        et = int(folder.parent.name)  # Environment Temperature (ET)
        x, y = load_xy(folder, normalized=self.normalized)
        return {
            "AOS": x,
            "GT": y,
            "ET": et,
            "IDX": idx,
        }
    
    def split(self, val_split: float, seed: Optional[int] = None):
        if not (0 < val_split < 1):
            raise ValueError("val_split must be between 0 and 1.")
        if seed is not None:
            random.seed(seed)
        
        n = len(self.folders)
        n_val = int(n * val_split)
        random.shuffle(self.folders)
        train_folders = self.folders[:-n_val]
        val_folders = self.folders[-n_val:]
        return AOSDataset(train_folders), AOSDataset(val_folders)


if __name__ == "__main__":
    meta_path = Path("/mnt/data/wildfire/IR/Batch-1.json")
    dataset = AOSDataset(None, meta_path, "area_07", 0.33)
    print(f"Dataset size: {len(dataset)}")
    dataset = dataset.split(0.1)[0]  # Get the training set
    print(f"Number of training samples: {len(dataset)}")

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print(batch["AOS"].shape, batch["GT"].shape, batch["Residual"].shape, batch["ET"].shape)
        break
