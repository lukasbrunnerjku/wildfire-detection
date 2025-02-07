import torch.multiprocessing as mp
from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass
from tqdm import tqdm

from .augmentation import build_dataset, process_dataset
from ..utils.image import build_subset_datasets


@dataclass
class Conf():
    datadir: str = "/mnt/data/wildfire/synthetic/generated"
    imagedirs: str = "/mnt/data/wildfire/synthetic/images/{:02d}_amb_temp"
    targetdirs: str = "/mnt/data/wildfire/synthetic/targets/{:02d}_amb_temp"
    start_amb_temp: int = 0
    step_amb_temp: int = 1
    stop_amb_temp: int = 35
    amb_temp: float = 9.0
    max_sun_temp_inc: float = 15.0
    upper_fire_thres_temp: float = 60.0
    max_amb_temp: float = 30.0
    max_amb_temp_inc: float = 5.0
    num_workers: int = 4


def parse_args():
    conf = OmegaConf.structured(Conf)
    file = OmegaConf.load("src/configs/multi_augment.yaml")
    conf = OmegaConf.merge(conf, file)
    return conf


if __name__ == "__main__":
    # NOTE: AttributeError: 'NoneType' object has no attribute 'dumps' <-- ignore this, seems to work fine!
    conf = parse_args()
    print(conf)

    amb_temps = [t for t in range(conf.start_amb_temp, conf.stop_amb_temp, conf.step_amb_temp)]

    dataset = build_dataset(Path(conf.datadir))

    num_total_samples = len(dataset)
    samples_per_subset = int(num_total_samples / len(amb_temps))
    assert num_total_samples % len(amb_temps) == 0

    datasets = build_subset_datasets(dataset, samples_per_subset)

    pool = mp.Pool(processes=conf.num_workers)

    pbar = tqdm(desc="Processing images...", total=num_total_samples)

    def progress_callback(num_images_processed: int):
        pbar.update(num_images_processed)
    
    for idx, new_amb_temp in enumerate(amb_temps):
        newdatadir = Path(conf.imagedirs.format(new_amb_temp))
        targetdir = Path(conf.targetdirs.format(new_amb_temp))

        # Heuristic, biomass can have a max. temp. of ie. 30, but if ie. the new amb. temp.
        # is already 30 degrees we can easily imagine that biomass could be a bit hotter than that.
        # Set the allowed max. amb. temp. to ie. 5 degrees hotter for high new amb. temp. settings.
        max_amb_temp = max(new_amb_temp + conf.max_amb_temp_inc, conf.max_amb_temp)  

        args = (
            datasets[idx], 0, newdatadir, targetdir,
            conf.amb_temp, conf.max_sun_temp_inc,
            new_amb_temp, max_amb_temp,
            conf.upper_fire_thres_temp,
            False,
        )
        pool.apply_async(process_dataset, args, callback=progress_callback).get()
    
