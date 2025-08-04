from torch.utils.data import Dataset
import json
from pathlib import Path
from typing import Optional
import random
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import seaborn as sns
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .utils import load_xy


class LargeAOSDataset(Dataset):
    def __init__(
        self,
        root: Path,
        folders: list[Path],  # Relative to "root"
        normalized: bool = False,
    ):
        self.root = root
        self.folders = folders
        self.normalized = normalized

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder: Path = self.root / self.folders[idx]  # */Batch-1/27/abc30f62e9ba47cab97c3ef3850a114d
        et = int(folder.parent.name)  # Environment Temperature (ET)
        x, y = load_xy(folder, normalized=self.normalized)
        return {
            "AOS": x,
            "GT": y,
            "ET": et,
            "IDX": idx,
        }


class LargeAOSDatsetBuilder:
    def __init__(self, root: Path, log_file: str = "log.txt"):
        """
        https://github.com/qubvel-org/segmentation_models.pytorch
        
        Folder structure as follows:
        
        root
            Batch-1
                0
                1
                ...
                30
                
            Batch-2
                0
                1
                ...
                30
            ...
            Batch-7
                0
                1
                ...
                30
                
        Inside folder ie. 0 we find following structure:
            0d96028a784948dcb2976cd134c84aa7
                images
                    0_550_pose_0_thermal.png
                    0_550_pose_0_thermal.pngmin_max_temp.txt
                    ...
                GT.tiff
                global_min_max_temp.txt
                integrall_0.png
                integrall_normalized_0.png
                label.png
                min_max_temp_GT.txt
                poses.txt
                scene_paramters.txt
            0ea7d6e418d54d3997cfd238d4a2fadb
                ...
            ...
        """
        super().__init__()
        self.root = root
        self.log_file = root / log_file
        # Create an empty log file (overwrite if exists)
        with open(self.log_file, "w") as fp:
            pass  # This creates an empty file
        
        train_csv_file = root / "train.csv"
        test_csv_file = root / "test.csv"
        
        if train_csv_file.exists() and test_csv_file.exists():
            print(f"Load: {train_csv_file} and {test_csv_file}")
            train_df = pd.read_csv(train_csv_file)
            test_df = pd.read_csv(test_csv_file)
        else:
            dfs = []
            batch_folders = [f for f in root.iterdir() if f.is_dir()]
            for batch_folder in batch_folders:
                csv_file = root / f"{batch_folder.name}.csv"
                
                if csv_file.exists():
                    print(f"Load: {csv_file}")
                    df = pd.read_csv(csv_file)
                else:
                    print(f"Working on: {batch_folder}")
                    df = self.build_table(batch_folder)
                    df.to_csv(csv_file)
                    
                dfs.append(df)
                
            # Merge all dataframes together
            df = pd.concat(dfs, ignore_index=True)
            
            train_df, test_df = self.stratified_split(df, bins=10)
            train_df.to_csv(train_csv_file)
            test_df.to_csv(test_csv_file)
            self.compare_distributions(train_df, test_df, bins=10)
        
        self.train_df = train_df
        self.test_df = test_df
    
    def get_dataset(
        self,
        split: str = "train",
        drop_uniform: bool = True,
        **kwargs,
    ) -> LargeAOSDataset:
        if split == "train":
            df = self.train_df
        elif split == "test":
            df = self.test_df
        else:
            raise ValueError(f"Unknown argument {split=}")
        
        if drop_uniform:
            n_total = 0
            folders = []
            for row in df.itertuples(index=False):
                n_total += 1
                if row.uniform_tex == 0:
                    folders.append(Path(row.aos_folder))
            print(f"Dropped uniform textures, reduced rows from {n_total} to {len(folders)}.")
        else:
            folders = [Path(p) for p in df["aos_folder"]]
            
        return LargeAOSDataset(self.root, folders, **kwargs)
        
    def compare_distributions(self, train_df: pd.DataFrame, test_df: pd.DataFrame, bins: int):
        # Set plotting style
        sns.set_theme(style="whitegrid")

        # Plot Environment Temperature
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(train_df["env_temp"], color="blue", label="Train", bins=bins)
        sns.histplot(test_df["env_temp"], color="orange", label="Test", bins=bins, alpha=0.6)
        plt.title("Environment Temperature Distribution")
        plt.xlabel("Temperature")
        plt.legend()

        # Plot Tree Density
        plt.subplot(1, 2, 2)
        sns.histplot(train_df["tree_density"], color="green", label="Train", bins=bins)
        sns.histplot(test_df["tree_density"], color="red", label="Test", bins=bins, alpha=0.6)
        plt.title("Tree Density Distribution")
        plt.xlabel("Tree Density")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.root / "train_test_dists.png")
        plt.close()
                
    def stratified_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        bins: int = 10,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Bin the temperature and forest density
        df["density_bin"] = pd.cut(df["tree_density"], bins=bins)
        df["temp_bin"] = pd.cut(df["env_temp"], bins=bins)
        
        # Create stratification key
        df["stratify_key"] = df["temp_bin"].astype(str) + "_" + \
                            df["uniform_tex"].astype(str) + "_" + \
                            df["density_bin"].astype(str)
        
        temp_counts = df["temp_bin"].value_counts().sort_index()
        density_counts = df["density_bin"].value_counts().sort_index()
        
        # Plot side-by-side bar plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Temperature plot
        axes[0].bar(temp_counts.index.astype(str), temp_counts.values, color="skyblue")
        axes[0].set_title("Environment Temperature Distribution")
        axes[0].set_xlabel("Temperature Bins (°C)")
        axes[0].set_ylabel("Number of Samples")
        axes[0].tick_params(axis="x", rotation=45)

        # Forest density plot
        axes[1].bar(density_counts.index.astype(str), density_counts.values, color="forestgreen")
        axes[1].set_title("Forest Density Distribution")
        axes[1].set_xlabel("Density Bins")
        axes[1].set_ylabel("Number of Samples")
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(self.root / "all_bins.png")
        plt.close()
        
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df["stratify_key"], 
            random_state=random_state,
        )
        return train_df, test_df
        
    def build_table(self, root: Path) -> pd.DataFrame:
        row_list = []
        temperature_folders = [f for f in root.iterdir() if f.is_dir()]
        for temperature_folder in tqdm(temperature_folders, desc="Processing..."):
            aos_folders = [f for f in temperature_folder.iterdir() if f.is_dir()]
            for aos_folder in aos_folders:
                min_max_file = aos_folder / "min_max_temp_GT.txt"
                param_file = aos_folder / "scene_parameters.txt"
                
                incomplete = False
                if min_max_file.exists() and param_file.exists():
                    # Uniform GT texture on the forest floor?
                    with open(min_max_file, "r") as fp:
                        line = fp.read()
                        min_temp, max_temp = line.split(" ")[0].split(",")
                        uniform_tex = int(min_temp == max_temp)
                
                    # Env. temp.? Tree density?
                    with open(param_file, "r") as fp:
                        """
                        Selected Env temp: 1
                        Selected thermal texture: 0057273.TIF
                        Number of trees per hectare: 117
                        Direct lightsun effect: 12
                        Added lightsun effect based on the Azimuth angle: 6
                        Azimuth angle (Alpha) in degrees: -62.437380577215286
                        Tree top temperature: 4°C / 277.15K
                        """
                        lines = fp.readlines()
                        env_temp = int(lines[0].split(":")[1].strip())
                        tree_density = int(lines[2].split(":")[1].strip())
                else:  # Some meta data is missing
                    incomplete = True
                    with open(self.log_file, "a") as f:
                        f.write(f"{aos_folder}\n")
                
                if not incomplete:
                    # Check for core data completeness
                    files_to_check = [
                        aos_folder / "GT.tiff",
                        aos_folder / "integrall_0.png",
                        aos_folder / "label.png",
                    ]
                    incomplete = sum(int(p.exists()) for p in files_to_check) != len(files_to_check)
                
                if incomplete:
                    print(f"Incomplete: {aos_folder=}")
                else:
                    row_list.append({
                        "aos_folder": str(aos_folder.relative_to(root.parent)),
                        "env_temp": env_temp,
                        "uniform_tex": uniform_tex,
                        "tree_density": tree_density,
                    })
                
        df = pd.DataFrame(
            row_list,
            columns=["aos_folder", "env_temp", "uniform_tex", "tree_density"],
        )
        print(df.head(3))
        return df
                

class AOSDataset(Dataset):
    def __init__(
        self,
        folders: Optional[list[Path]] = None,
        meta_path: Optional[Path] = None,
        key: str = "score",
        threshold: float = 0.33,
        normalized: bool = False,
        # NOTE: If an argument is added here, do so in the ".split" method constructors!
    ):
        self.key = key
        self.threshold = threshold
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
        
        # NOTE: Add new argument here!
        train_ds = AOSDataset(
            train_folders,
            None,
            self.key,
            self.threshold,
            self.normalized,
        )
        val_ds = AOSDataset(
            val_folders,
            None,
            self.key,
            self.threshold,
            self.normalized,
        )
        return train_ds, val_ds 
    

class MeanStdCalculator:

    def __init__(self):
        self.psum    = torch.tensor([0.0], dtype=torch.float64)
        self.psum_sq = torch.tensor([0.0], dtype=torch.float64)
        self.n_pixel = torch.tensor([0.0], dtype=torch.float64)

    def update(self, x: Tensor):
        B, H, W = x.shape
        self.psum    += x.sum()
        self.psum_sq += (x ** 2).sum()
        self.n_pixel += B * H * W

    def compute(self) -> tuple[Tensor, Tensor]:
        """
        mean := E[X]
        std := sqrt(E[X^2] - (E[X])^2)
        """
        mean = self.psum / self.n_pixel
        var  = (self.psum_sq / self.n_pixel) - (mean ** 2)
        std  = torch.sqrt(var)
        return mean, std
    
    
def get_mean_std(dl: DataLoader):
    aos_calc = MeanStdCalculator()
    gt_calc = MeanStdCalculator()
    res_calc = MeanStdCalculator()

    for batch in tqdm(dl, "Calculating mean / std..."):
        aos = batch["AOS"]  # BxHxW
        gt = batch["GT"]  # BxHxW
        res = gt - aos

        aos_calc.update(aos)
        gt_calc.update(gt)
        res_calc.update(res)
        
    aos_mean, aos_std = aos_calc.compute()    
    gt_mean, gt_std = gt_calc.compute()    
    res_mean, res_std = res_calc.compute()    

    return {
        "AOS_mean": float(aos_mean),
        "AOS_std": float(aos_std),
        "GT_mean": float(gt_mean),
        "GT_std": float(gt_std),
        "RES_mean": float(res_mean),
        "RES_std": float(res_std),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/data/wildfire/IR/root"))
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--drop_uniform", action="store_true")
    args = parser.parse_args()
    
    builder = LargeAOSDatsetBuilder(args.root)
    train_ds = builder.get_dataset("train", normalized=args.normalized, drop_uniform=args.drop_uniform)
    test_ds = builder.get_dataset("test", normalized=args.normalized, drop_uniform=args.drop_uniform)
    
    mean_std_file = args.root / f"mean_std_n{int(args.normalized)}_du{int(args.drop_uniform)}.json"
    if not mean_std_file.exists():
        train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=16)
        train_stats = get_mean_std(train_dl)
        with open(mean_std_file, "w") as fp:
            json.dump(train_stats, fp, indent=4)
    
    train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=2, shuffle=True)

    for batch in train_dl:
        print(batch["AOS"].shape, batch["GT"].shape, batch["IDX"].shape, batch["ET"].shape)
        break
    
    for batch in test_dl:
        print(batch["AOS"].shape, batch["GT"].shape, batch["IDX"].shape, batch["ET"].shape)
        break
    
    # ~70% of data remains (drop_uniform=True)
    """
    Dropped uniform textures, reduced rows from 25946 to 18186.
    Dropped uniform textures, reduced rows from 6487 to 4547.
    torch.Size([2, 512, 512]) torch.Size([2, 512, 512]) torch.Size([2]) torch.Size([2])
    torch.Size([2, 512, 512]) torch.Size([2, 512, 512]) torch.Size([2]) torch.Size([2])
    """
    
    
# if __name__ == "__main__":
#     meta_path = Path("/mnt/data/wildfire/IR/Batch-1.json")
#     dataset = AOSDataset(None, meta_path, "area_07", 0.33)
#     print(f"Dataset size: {len(dataset)}")
#     dataset = dataset.split(0.1)[0]  # Get the training set
#     print(f"Number of training samples: {len(dataset)}")

#     from torch.utils.data import DataLoader

#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#     for batch in dataloader:
#         print(batch["AOS"].shape, batch["GT"].shape, batch["IDX"].shape, batch["ET"].shape)
#         break
