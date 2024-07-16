import wandb

sweep_configuration = {
    'method': 'bayes',
    'metric': {'goal': 'maximize', 'name': 'average_test_jaccard_index'},
    'parameters': {
        'LEARNING_RATE': {'max': 0.01, 'min': 0.0001}
    },
    'program': 'train.py',
    'command': [
        '${env}',
        'python',
        '${program}',
        'configs.config',
        '--tune',
        '--num_trials',
        '2'
    ]
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project='cmap_train')
print(f"Sweep ID: {sweep_id}")