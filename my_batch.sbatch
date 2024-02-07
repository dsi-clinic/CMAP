#!/bin/bash

#SBATCH --job-name=g005
#SBATCH --mail-user=rubensteinm@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/rubensteinm/slurm/out/%j.%N.stdout
#SBATCH --error=/home/rubensteinm/slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/rubensteinm/slurm
#SBATCH --partition=general
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

export PATH="/home/rubenstein/miniconda/bin:$PATH"

pip install -r /net/projects/cmap/requirements/requirements.txt

conda activate cmap

python train.py $SLURM_ARRAY_TASK_ID