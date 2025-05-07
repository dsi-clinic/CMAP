#!/bin/bash -l
#SBATCH --mail-user=gregoryc25@cs.uchicago.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=sam2-ft-ddp
#SBATCH --output=/home/gregoryc25/slurm/out/%j.stdout
#SBATCH --error=/home/gregoryc25/slurm/out/%j.stderr
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=4                # 4 total DDP processes
#SBATCH --gres=gpu:4              # 1 GPU per DDP process
#SBATCH --cpus-per-task=4         # 4 CPU threads per rank
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/gregoryc25/CMAP/segment_anything/utils

# Load shell hooks for micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate cmap

# Set master address and unique port for DDP
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((12000 + RANDOM % 10000))

# Launch DDP training
srun --cpu-bind=none python -m torch.distributed.run \
     --nproc_per_node=$SLURM_NTASKS \
     /home/gregoryc25/CMAP/segment_anything/utils/train_sam.py \
     --image-dir /net/projects/cmap/data/KC-images \
     --mask-dir  /net/projects/cmap/data/KC-masks/single-band-masks \
     --mask-prefix mask_ \
     --checkpoint /home/gregoryc25/CMAP/segment_anything_source_code/sam_vit_h.pth \
     --output /home/gregoryc25/CMAP/segment_anything/fine_tuned_ddp.pth \
     --epochs 5 \
     --batch-size 1 \
     --accum-steps 4 \
     --lr 1e-4

