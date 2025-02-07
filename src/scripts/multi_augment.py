import multiprocessing as mp
import subprocess
from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass


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


def parse_args():
    conf = OmegaConf.structured(Conf)
    file = OmegaConf.load("src/configs/multi_augment.yaml")
    conf = OmegaConf.merge(conf, file)
    return conf


def run_script(script_name: str, args: list[str]):
    """Run a script with given arguments in a separate process."""
    cmd = ["python -m", script_name] + args  # Build command
    process = subprocess.Popen(cmd)  # Start process
    process.wait()  # Wait for completion


if __name__ == "__main__":
    conf = parse_args()
    print(conf)

    ts = [t for t in range(conf.start_amb_temp, conf.stop_amb_temp, conf.step_amb_temp)]
        
    # conf.amb_temp
    # conf.max_sun_temp_inc
    # conf.upper_fire_thres_temp

    # imagedir = conf.imagedirs.format(new_amb_temp)
    # targetdir = conf.targetdirs.format(new_amb_temp)

# # Define scripts and their respective command-line arguments
# scripts = [
#     ("script1.py", ["arg1", "arg2"]),
#     ("script2.py", ["argA", "argB"]),
#     ("script3.py", ["123", "456"]),
# ]



# if __name__ == "__main__":
#     processes = []

#     for script, args in scripts:
#         p = multiprocessing.Process(target=run_script, args=(script, args))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()  # Wait for all processes to finish

#     print("All scripts have finished execution.")