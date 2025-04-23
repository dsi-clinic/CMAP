#!/bin/bash -l
#SBATCH --mail-user=gregoryc25@cs.uchicago.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=sam2-ft-ddp
#SBATCH --output=/home/gregoryc25/slurm/out/%j.stdout
#SBATCH --error=/home/gregoryc25/slurm/out/%j.stderr
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=4          # total DDP ranks
#SBATCH --gres=gpu:4        # one GPU per rank
#SBATCH --cpus-per-task=4   # threads per rank
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Set up DDP rendezvous
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29505

# Activate your environment
eval "$(micromamba shell hook -s bash)"
micromamba activate cmap

# Launch 4 processes, disable Slurm CPU binding
srun --cpu-bind=none python -m torch.distributed.run \
     --nproc_per_node=$SLURM_NTASKS \
     $(dirname "$0")/train_sam.py \
       --image-dir /net/projects/cmap/data/KC-images \
       --mask-dir  /net/projects/cmap/data/KC-masks/single-band-masks \
       --checkpoint /home/gregoryc25/CMAP/segment_anything_source_code/sam_vit_h.pth \
       --output fine_tuned_ddp.pth \
       --epochs 5 \
       --batch-size 1 \
       --accum-steps 4 \
       --lr 1e-4
