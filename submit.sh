#!/bin/bash -l
#
#SBATCH --mail-user=xiaoyue1@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/xiaoyue1/slurm/out/%j.%N.stdout
#SBATCH --error=/home/xiaoyue1/slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/xiaoyue1/slurm
#SBATCH --partition=general
#SBATCH --job-name=cmap
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

source /home/xiaoyue1/miniconda3/bin/activate cmap

export PATH="/home/xiaoyue1/miniconda/bin:$PATH"

cd /home/xiaoyue1/2024-winter-cmap

python train.py configs.config --experiment_name patch_size_512 --num_trial 5
