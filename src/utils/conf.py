from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    folder: Optional[str] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    image_size: int = 512
    num_crops: int = 2
    num_fid_samples: int = 512
    do_resize: bool = False
    extention: str = "TIF"
