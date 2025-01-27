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
python -m src.augmentation 
```

Train the LDM with

```
python -m src.ldm.train vqvae_checkpoint=path/to/vqvae.ckpt data.folder=path/to/tif/images data.mean=somevalue data.std=somevalue logdir=path/to/output/of/training
```

Example:

```
python -m src.ldm.train vqvae_checkpoint=/mnt/data/wildfire/runs/VQVAE/version_4/checkpoints/step=39500-fid_score=4.704.ckpt data.folder=/mnt/data/wildfire/imgs1 data.mean=2.7935 data.std=5.9023 logdir=/mnt/data/wildfire/runs
```

## Running FID Analytics

What FID values to expect taking the ground truth as lower bound

```
python -m src.utils.fid
```
