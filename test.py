from dataclasses import dataclass, Field, field
from omegaconf import OmegaConf

@dataclass
class SubConfig:
    x: int = 3

@dataclass
class Config:
    a: int
    b: str = "ab"
    subc: SubConfig = field(default_factory=SubConfig)


if __name__ == "__main__":
    conf = OmegaConf.structured(Config)
    args = OmegaConf.from_cli()
    print(conf, args)

    conf = OmegaConf.merge(conf, args)
    print(conf)
    print(conf.a)