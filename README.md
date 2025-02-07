# Master Thesis on Wildfire-Detection

Augment thermal data for different environment temperatures, train a VQVAE and LDM to generate new thermal images, and automatically annotate thermal data for semantic segmentation.

## Installation

Either make a virtual python environment like 

```
py -3.12 -m venv .venv --upgrade-deps
```
or by using conda

```
conda create --name wildfire python=3.12
```

Install the package dependencies

```
pip install -r requirements.txt
```

## Running Code

The original thermal data mean and standard deviation is 2.7935 and 5.9023, for the
augmented thermal data statistics can be calcualted with

```
python -m src.scripts.statistics path/to/tif/images
```

Example 

```
python -m src.scripts.statistics /mnt/data/wildfire/imgs1
```

Train the VQVAE with

```
python -m src.vqvae.train data.folder=path/to/tif/images data.mean=somevalue data.std=somevalue logdir=path/to/output/of/training
```

Example:

```
python -m src.vqvae.train data.folder=/mnt/data/wildfire/imgs1 data.mean=2.7935 data.std=5.9023 logdir=/mnt/data/wildfire/runs
```

Monitor the training progress via tensorboard with ie.

```
tensorboard --logdir="/mnt/data/wildfire/runs"
```

Augment the thermal images by simulating new a environment temperature with

```
python -m src.scripts.augmentation path/to/tif/images path/to/save/targets --newdatadir path/to/augmented/tif/images
```

When targets of existing tif images should be generated run the above command without: --newdatadir path/to/augmented/tif/images

Example:

```
python -m src.scripts.augmentation /mnt/data/wildfire/imgs1 /mnt/data/wildfire/targets --newdatadir /mnt/data/wildfire/new_imgs1 --amb_temp 9 --max_sun_temp_inc 15 --new_amb_temp 20 --max_amb_temp 30 --upper_fire_thres_temp 60
```

Train the LDM with

```
python -m src.ldm.train vqvae_checkpoint=path/to/vqvae.ckpt data.folder=path/to/tif/images data.mean=somevalue data.std=somevalue logdir=path/to/output/of/training
```

Example:

```
python -m src.ldm.train vqvae_checkpoint=/mnt/data/wildfire/runs/VQVAE/version_4/checkpoints/step=39500-fid_score=4.704.ckpt data.folder=/mnt/data/wildfire/imgs1 data.mean=2.7935 data.std=5.9023 logdir=/mnt/data/wildfire/runs
```

Generating novel images with the LDM with

```
OMP_NUM_THREADS=16 torchrun --nproc-per-node 4 -m src.scripts.generate /mnt/data/wildfire/synthetic/generated /mnt/data/wildfire/runs/LDM/version_5/checkpoints/step
=19000-fid_score=8.218.ckpt 3500 --batch_size 16
```

We assign each of the 4 GPUs 16 CPUs and generate 3500 images using a batch size of 32 per GPU to run the inference code. The output is saved to /mnt/data/wildfire/synthetic/generated, in a next step we can produce images with different environment temperatures by a heuristic augmentation strategy. With a batch size of 16 the GPU memory consumption is already a bit over 24 Gigabyte on the GPU with rank 0 (18 Gigabyte on the others).

Augmenting the generated images automatically for a range of specified ambient temperatures can be done with

```
python -m src.scripts.multi_augment
```



## Running Analytics and Visualizations

What FID values to expect taking the ground truth as lower bound

```
python -m src.utils.fid
```

Visualize the effect of temperature augmentation with

```
python -m src.scripts.make_gif /mnt/data/wildfire/misc /mnt/data/wildfire/imgs1/DJI_20231017140640_0267_T.TIF 0 30
```

Make tone mapped image from *.TIF files with

```
python -m src.scripts.tonemap /mnt/data/wildfire/new_imgs1 /mnt/data/wildfire/tonemapped
```
