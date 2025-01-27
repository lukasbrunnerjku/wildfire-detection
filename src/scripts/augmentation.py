import torch
from torch import Tensor


def amb_temp_aug(
    img: Tensor,
    amb_temp: float,
    max_sun_temp_inc: float,
    new_amb_temp: float,
    max_amb_temp: float,
):
    delta_amb = new_amb_temp - amb_temp
    new_img = img.clone()
    fire_thres_temp = amb_temp + max_sun_temp_inc
    fire_mask = img >= fire_thres_temp

    # Increas non-fire pixels >0 degree celcius by the temp. delta
    # and make sure the max. ambient temp. condition is not violated.
    amb_mask = torch.logical_and(~fire_mask, new_img > 0)
    new_img[amb_mask] = torch.clip(new_img[amb_mask] + delta_amb, max=max_amb_temp)

    # Set fire-pixels < new ambient temp. to new ambient temp.
    new_img[torch.logical_and(fire_mask, new_img < new_amb_temp)] = new_amb_temp
    return new_img


def main(
    amb_temp: float,
    max_sun_temp_inc: float,
    new_amb_temp: float,
    max_amb_temp: float,
):
    new_img = amb_temp_aug(
        img,
        amb_temp,
        max_sun_temp_inc,
        new_amb_temp,
        max_amb_temp,
    )
    new_fire_thres_temp = new_amb_temp + max_sun_temp_inc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", help="Directory to .TIF files")
    parser.add_argument("newdatadir", help="Directory to save new .TIF files")
    parser.add_argument("targetdir", help="Directory to save new targets")
    parser.add_argument("--amb_temp", type=float, default=9)
    parser.add_argument("--max_sun_temp_inc", type=float, default=15)
    parser.add_argument("--new_amb_temp", type=float, default=20)
    parser.add_argument("--max_amb_temp", type=float, default=30)
    main()