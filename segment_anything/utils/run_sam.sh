#!/bin/bash -l
#
#SBATCH --mail-user=gregoryc25@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/gregoryc25/slurm/out/%j.%N.stdout
#SBATCH --error=/home/gregoryc25/slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/gregoryc25/slurm
#SBATCH --partition=general
#SBATCH --job-name=cmap-array
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-6

# Initialize micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate cmap

# Adjust user as needed
cd /home/gregoryc25/CMAP/segment_anything/utils

# Run segment_kc.py with array-based subset logic
python segment_kc.py \
    --total_subsets=7 \
    --subset_index=${SLURM_ARRAY_TASK_ID} \
    --max_samples=-1

# NOTE:
# - This job will produce partial CSVs named
#   per_class_ious_subset_{0..6}.csv
#   inside /home/gregoryc25/CMAP/segment_anything/kc_sam_outputs/kc_sam_run_<JOBID>/
# - It may also produce up to 20 debug images per task in a 'plots/' subdir.
# - No final aggregator logic is here.
