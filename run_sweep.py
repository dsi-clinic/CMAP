import subprocess
import wandb

sweep_configuration = {
    'method': 'bayes',
    'metric': {'goal': 'maximize', 'name': 'average_test_jaccard_index'},
    'parameters': {
        # 'COLOR_BRIGHTNESS': {'values': [0.2, 0.3, 0.4, 0.5]},
        # 'GAUSSIAN_BLUR_SIGMA': {'values': [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]},
        # 'GAUSSIAN_BLUR_KERNEL': {'values': [(3,3), (5,5), (7,7)]},
        'LEARNING_RATE': {'min': 0.0001, 'max': 0.01},
        # 'ATTENTION_HEAD_DIM': {'values': [8, 16, 32]},
        # 'LAYERS_PER_BLOCK': {'values': [1, 2, 3]},
        # 'NORM_NUM_GROUPS': {'values': [16, 32, 64]},
        # 'NORM_EPS': {'values': [1e-5, 1e-6, 1e-7]}
    }
}

sweep_id = wandb.sweep(sweep_configuration, project="cmap_train")

def train():
    subprocess.call(["python", "train.py", "configs.config", "--tune", "--num_trials", "2"])

wandb.agent(sweep_id, function=train)