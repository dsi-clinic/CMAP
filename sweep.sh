#!/bin/bash

# Assuming the sweep name or ID is passed as the first argument to the script
# agent_cmd="wandb agent $1 --count 10"
agent_cmd="wandb agent cmap-2024/cmap/eundnj2o --count 10"

# Using SLURM_JOB_ID to create unique log filenames
mkdir -p "sweep_logs/"
CUDA_VISIBLE_DEVICES=0 nohup $agent_cmd >& "sweep_logs/sweep_0_${HOSTNAME}_${SLURM_JOB_ID}.log" &
CUDA_VISIBLE_DEVICES=1 nohup $agent_cmd >& "sweep_logs/sweep_1_${HOSTNAME}_${SLURM_JOB_ID}.log" &
# CUDA_VISIBLE_DEVICES=2 nohup $agent_cmd >& "sweep_logs/sweep_2_${HOSTNAME}_${SLURM_JOB_ID}.log" &
# CUDA_VISIBLE_DEVICES=3 nohup $agent_cmd >& "sweep_logs/sweep_3_${HOSTNAME}_${SLURM_JOB_ID}.log" &