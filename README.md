# 2024-winter-cmap

## Project Background

No comprehensive inventory of stormwater storage and green infrastructure (GI) assets exists across northeastern Illinois. Understanding the location of these assets is critical to ensuring proper maintenance as well as building a better understanding of the potential impacts to water quality and stormwater management. An inventory could help county and municipal stormwater engineers, public works officials, and others ensure proper maintenance. The data could also inform the development of watershed-based plans and resilience plans.

The Chicago Metropolitan Agency for Planning (CMAP) is interested in using deep learning to map and identify locations of stormwater storage and other related geographic features throughout Chicago and the surrounding area.
To begin the project, CMAP has provided labeled geographic features in Kane County, Illinois (provided by Kane County), to be used to create a predictive deep learning model.
The code in this repo does a few things:
1. Creates masks of geographic features across Kane County.
2. Will train and test various predictive deep learning models on surrounding geographies.
3. Will apply Kane County data to identify stormwater basins in other Illinois counties.

## Project Goals

There are several tasks associated with this project:

1. Improve climate resiliency in northeastern Illinois with deep learning for mapping stormwater and green infrastructure from aerial data.
2. Develop deep learning models for aerial imaging data, targeting green infrastructure and stormwater areas.
3. Train a model to identify different types of locations (for example, wet ponds, dry-turf bottom, dry-mesic prairie, and constructed wetland detention basins) and then use this model to identify other areas of the region with these attributes.

This will be accomplished within the following pipeline structure:
1. Masks of shape data are created for Kane County stormwater structures (`preproc_kc.py`) and if necessary, original NAIP images are downloaded (`utils/get_naip_images.py`).
2. A training loop (`train.py`) takes in configurations (`configs/dsi.py`) and is assigned to the cluster (.job), utilizing the model defined `utils/model.py` and the custom Raster Dataset defined in `utils/kc.py`.

## Usage

Before running the repo (see details below) you will need to do the following:
1. Install make if you have not already done so.
2. Create and initiate a cmap specific conda environment using the following steps:
    1. Install miniconda:
    ```
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    ```
    2. Create environment:
    ```
    conda create -y --name cmap python=3.10
    conda activate cmap
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
    pip install -r /home/YOUR_USERNAME/2024-winter-cmap/requirements.txt
    ```

### Slurm

For more information about how to use Slurm, please look at the information [here](https://github.com/uchicago-dsi/core-facility-docs/blob/main/slurm.md).

To run this repo on the Slurm cluster after setting up your conda environment, you can use the following submit script to run a training loop:
```
#!/bin/bash -l
#
#SBATCH --mail-user=YOUR_USERNAME@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/YOUR_USERNAME/slurm/out/%j.%N.stdout
#SBATCH --error=/home/YOUR_USERNAME/slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/YOUR_USERNAME/slurm
#SBATCH --partition=general
#SBATCH --job-name=cmap
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

source /home/YOUR_USERNAME/miniconda3/bin/activate cmap

export PATH="/home/YOUR_USERNAME/miniconda/bin:$PATH"

cd /home/YOUR_USERNAME/2024-winter-cmap

python train.py configs.dsi [--experiment_name <ExperimentName>] [--aug_type <aug>] [--split <split>] $SLURM_ARRAY_TASK_ID
```

Or, to run in an interactive session:
```
srun -p general --pty --cpus-per-task=8 --gres=gpu:1 --mem=128GB -t 0-06:00 /bin/bash

conda activate cmap

cd /home/YOUR_USERNAME/2024-winter-cmap

python train.py configs.dsi [--experiment_name <ExperimentName>] [--aug_type <aug>] [--split <split>]
```

## Git Usage

Before pushing changes to git, ensure that you're running `pre-commit run --all` to check your code against the linter.

## Repository Structure

### utils

Project python code. Contains various utility functions and scripts which support the main functionalities of the project and are designed to be reusable. 

### notebooks

Contains short, clean notebooks to demonstrate analysis. Documentation and descriptions included in the [README](notebooks/README.md) file.

### data

Contains details of acquiring all raw data used in repository. If data is small (<50MB) then it is okay to save it to the repo, making sure to clearly document how to the data is obtained.

If the data is larger than 50MB than you should not add it to the repo and instead document how to get the data in the README.md file in the data directory. 

Source attribution and descriptions included in the [README](data/README.md) file.

### output

Contains example model output images.

## Preliminary Results
The below results were obtained with these specifications:
* Classes: "POND" "WETLAND" "DRY BOTTOM - TURF" "DRY BOTTOM - MESIC PRAIRIE"
* Batch size: 16
* Patch size: 512
* Learning rate: 1E-5
* Number of workers: 8
* Epochs: 30 (maximum; early termination feature has been turned on)

| Model | Backbone | Weights | Final IoU | Final Loss |
| ----------- | ----------- | ----------- | ----------- | ----------- | 
| deeplabv3+ | resnet50 | imagenet | 0.631 | 0.246 |
| deeplabv3+ | resnet50 | ssl | 0.548 | 0.232 |
| deeplabv3+ | resnet50 | swsl | 0.558 | 0.233 |
| unet | resnet50 | imagenet | 0.515 | 0.226 |
| unet | resnet50 | ssl | 0.560 | 0.209 |
| unet | resnet50 | swsl | 0.590 | 0.226 |
| unet | resnet18 | LANDSAT_ETM_SR_SIMCLR | 0.530 | 0.264 |
| unet | resnet18 | LANDSAT_ETM_SR_MOCO | 0.519 | 0.227 |

### example outputs
The model can detect ponds fairly accurately:
![output_image1](/output/example_images/DL_ResNet50_imagenet_v1/epoch-14/test_sample-14.0.0.png)
![output_image2](/output/example_images/DL_ResNet50_imagenet_v1/epoch-14/test_sample-14.0.7.png)
![output_image3](/output/example_images/DL_ResNet50_imagenet_v1/epoch-14/test_sample-14.0.8.png)

There needs to be some tweaks for the model to better identify wetlands and dry bottom turf stormwater infrastructure:
![output_image4](/output/example_images/DL_ResNet50_imagenet_v1/epoch-14/test_sample-14.0.2.png)
![output_image5](/output/example_images/DL_ResNet50_imagenet_v1/epoch-14/test_sample-14.0.5.png)

There also needs to be adjustments made to the model to account for false positives:
![output_image6](/output/example_images/DL_ResNet50_imagenet_v1/epoch-14/test_sample-14.1.6.png)
![output_image7](/output/example_images/DL_ResNet50_imagenet_v1/epoch-14/test_sample-14.1.10.png)

## Collaborators
- Matthew Rubenstein - rubensteinm@uchicago.edu
- Spencer Ellis - sjne@uchicago.edu
- Tamami Tamura - tamamitamura@uchicago.edu
- Mingyan Wang - mingyan@uchicago.edu
- Miao Li - mli628@uchicago.edu
