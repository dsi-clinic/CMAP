#!/bin/bash -l
#

#SBATCH --output=/net/scratch/ijain1/diffusion_sat_checkpoints/4channels_1.stdout
#SBATCH --error=/net/scratch/ijain1/diffusion_sat_checkpoints/4channels_1.stderr
#SBATCH --chdir=/home/ijain1/slurm
#SBATCH --partition=general
#SBATCH --job-name=cmap4channels_1
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

source /home/ijain1/miniconda2/bin/activate cmap

export PATH="/home/ijain1/miniconda/bin:$PATH"

cd /home/ijain1/2024-winter-cmap

python train.py configs.baseline_config --experiment_name 4channels_1
