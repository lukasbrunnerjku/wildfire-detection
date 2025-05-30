from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    # Copy files from root2 to root1
    parser.add_argument("--root1", default=r"\\wsl.localhost\Ubuntu-24.04\home\lbrunn\projects\irdata\data\Batch-1")  # "C:/IR/data/Batch-1"
    parser.add_argument("--root2", default="C:/Users/lbrunn/Downloads/Correct/Batch-1")
    args = parser.parse_args()
    
    root1 = Path(args.root1)
    folder_paths = list(root1.iterdir())
    subfolder_paths = []
    for folder_path in folder_paths:
        subfolder_paths.extend(list(folder_path.iterdir()))
    
    subfolder_paths = [d.relative_to(root1) for d in subfolder_paths]
    
    root2 = Path(args.root2)
    
    for subfolder_path in tqdm(subfolder_paths, desc="Processing..."):
        folder1 = (root1 / subfolder_path)
        folder2 = (root2 / subfolder_path)
        
        image1 = folder1 / "integrall_0.png"
        image2 = folder2 / "integrall_normalized_0.png"
        
        # From folder2 to folder1
        shutil.copyfile(image2, folder1 / "integrall_normalized_0.png")
        shutil.copyfile(folder2 / "global_min_max_temp.txt", folder1 / "global_min_max_temp.txt")
        
        # image1 = plt.imread(image1)
        # image2 = plt.imread(image2)
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(image1)
        # ax2.imshow(image2)
        # plt.show()
        # plt.close(fig)
    