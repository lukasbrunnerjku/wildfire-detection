import zipfile
from pathlib import Path


def extract(zip_file: Path, target_dir: Path):
    print(f"Extracting {zip_file} ...")
    with zipfile.ZipFile(str(zip_file), "r") as zip_ref:
        zip_ref.extractall(str(target_dir))


def find_zip_files(directory: Path) -> list:
    zip_files = list(directory.glob("*.zip"))
    if len(zip_files) > 0:
        print(f"Found {len(zip_files)} zip files: {zip_files[:2]} ... {zip_files[-1]}")
    else:
        print(f"No more zip files found in: {directory}")
    return zip_files

    
def extract_zip_recursive(zip_path: Path, extract_to: Path):
    if zip_path.is_dir():
        # Look into this directory
        new_dir = zip_path
    else:
        # Initial extraction
        extract(zip_path, extract_to)
    
        # Path to extracted folder
        new_dir = zip_path.with_suffix("")
    
    # Search extracted folder for .zip files, and extract
    new_zip_files = find_zip_files(new_dir)
    for new_zip_file in new_zip_files:
        extract_zip_recursive(new_zip_file, new_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_file", type=Path, help="yourfile.zip or input_folder")
    parser.add_argument("outdir", type=Path, help="output_folder")
    args = parser.parse_args()
    
    extract_zip_recursive(args.zip_file, args.outdir)
