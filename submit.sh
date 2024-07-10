#!/bin/bash -l
#

#SBATCH --output=/net/scratch/ijain1/diffusion_sat_checkpoints/%j.%N.stdout
#SBATCH --error=/net/scratch/ijain1/diffusion_sat_checkpoints/%j.%N.stderr
#SBATCH --chdir=/home/ijain1/slurm
#SBATCH --partition=general
#SBATCH --job-name=cmap
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

source /home/ijain1/miniconda3/bin/activate diffsat

export PATH="/home/ijain1/miniconda/bin:$PATH"

cd /home/ijain1/diffusers/examples/text_to_image

accelerate launch --multi_gpu --num_processes=4 train_text_to_image.py \
  --resume_from_checkpoint=/net/scratch/ijain1/diffusion_sat_final_model/checkpoint-10500 \
  --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
  --dataset_name=lambdalabs/naruto-blip-captions \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler=constant --lr_warmup_steps=0 \
  --output_dir=/net/scratch/ijain1/diffusion_sat_final_model \
  --checkpointing_steps=500
