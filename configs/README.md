# Running Hyperparameter Sweep with SLURM and Weights & Biases
This guide provides instructions to set up and run a hyperparameter sweep using SLURM and Weights & Biases (W&B) with a micromamba environment.

## Prerequisites
Create a Weights & Biases Account
Sign up for a free W&B account at https://wandb.ai if you don’t have one already.

Install Micromamba
Ensure micromamba is installed, and your environment is set up with all dependencies.

1. Install Weights & Biases (wandb)
Install W&B in your micromamba environment by running:

'''
micromamba install -c conda-forge wandb
''' 
2. Log in to W&B: Log into your W&B account by running:
'''
wandb login
'''

## Initialize the Sweep in W&B
Run the following command to create the sweep in W&B. This command will return a sweep_id, which you will need to start the sweep.
'''
wandb sweep /home/<YOUR-USERNAME>/CMAP/configs/sweep_config.yml
'''
Change YOUR-USERNAME to your username. Note the sweep_id that W&B provides, as it will look something like <your_project>/<sweep_id>.

## Update and Run the SLURM Job Script
Edit the sweep.job file to set up the SLURM job, activate your environment, and start the W&B agent. Replace <dsi-clinic-cmap/CMAP/6l5xz4qw> with your actual <sweep_id>.

## Submit the job to SLURM using:
'''
sbatch sweep.job
''' 

