#!/bin/bash -l
#
#SBATCH --mail-user=gregoryc25@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/gregoryc25/slurm/out/%j.%N.stdout
#SBATCH --error=/home/gregoryc25/slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/gregoryc25/slurm
#SBATCH --partition=general
#SBATCH --job-name=cmap-aggregate
#SBATCH --time=0:30:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=2

eval "$(micromamba shell hook -s bash)"
micromamba activate cmap

# Move to code directory
cd /home/gregoryc25/CMAP

# We call aggregator_kc.py to combine partial CSVs into one final CSV
python aggregator_kc.py
