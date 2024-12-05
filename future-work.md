
# Future Work

## Hyperparameter Tuning

### Fix Errors for Running a Sweep Using a SLURM Job Script
Currently, errors are encountered while running SLURM jobs.  
Example:  
```
3964     self._accessor.mkdir(self, mode)
3965 FileNotFoundError: [Errno 2] No such file or directory: '/net/projects/cmap/workspaces/xiaoyue1/20241202-092952_trial_6/train-images/epoch-1'
```

These errors indicate issues with directory cleanup and recreation during repeated runs. Once resolved, follow these steps to run a sweep using a SLURM job script:

1. **Edit the `sweep.job` File**:
   - Replace `<YOUR-USERNAME>` with your username.
   - Replace `<W&B_organization_name>` and `<sweep_id>` with the appropriate values from your W&B project.

2. **Submit the Job and Monitor Its Status**:
   ```bash
   sbatch sweep.job
   squeue -u <YOUR-USERNAME>
   ```

---

### Future Tuning After Incorporating Additional Data Sources

1. **Initial Experiment Design Using Bayesian Search**

For Bayesian search, the number of initial experiments depends on the parameter space size and the desired balance between exploration and exploitation.

- **Case with 3 Parameters**:
  Parameters:
  ```yaml
  batch_size:
    values: [8, 16, 32]
  learning_rate:
    values: [1e-4, 1e-5, 1e-6]
  patch_size:
    values: [256, 512, 768]
  ```
  The total search space is \(3 	imes 3 	imes 3 = 27\). Start with **10-15 initial experiments** for sufficient exploration of the parameter space.

- **Case with 6 Parameters**:
  Parameters:
  ```yaml
  batch_size:
    values: [8, 16, 32]
  learning_rate:
    values: [1e-4, 1e-5, 1e-6]
  patch_size:
    values: [256, 512, 768]
  color_brightness:
    values: [0.2, 0.3, 0.4, 0.5]
  gaussian_blur_sigma:
    values:
      - [0.1, 0.2]
      - [0.3, 0.4]
      - [0.5, 0.6]
  gaussian_blur_kernel:
    values:
      - [3,3]
      - [5,5]
      - [7,7]
  ```
  The total search space is \(3 	imes 3 	imes 3 	imes 4 	imes 3 	imes 3 = 972\). Start with **30-50 initial experiments** to adequately sample the high-dimensional space.

Bayesian search will iteratively refine the parameter selection based on previous results, ensuring efficient exploration of the search space.


2. **Run Sweeps Using Multiple Agents**  
   Running multiple agents in parallel can significantly reduce testing time. Update the `sweep.job` script to deploy multiple agents simultaneously:

   - Specify the number of tasks (agents) and GPUs in the SLURM job script:
     ```bash
     #SBATCH --ntasks=3
     #SBATCH --gres=gpu:3
     ```
   - Launch agents using `srun` for proper resource allocation:
     ```bash
     srun --exclusive --ntasks=1 --gres=gpu:1 --cpus-per-task=10           wandb agent username/project_name/sweep_id --count 10 &
     srun --exclusive --ntasks=1 --gres=gpu:1 --cpus-per-task=10           wandb agent username/project_name/sweep_id --count 10 &
     srun --exclusive --ntasks=1 --gres=gpu:1 --cpus-per-task=10           wandb agent username/project_name/sweep_id --count 10 &
     ```

3. **Switch to Continuous Search After Narrowing Parameter Ranges**  
   Currently, the sweep configuration uses discrete search (e.g., for batch size: `[8, 16, 32]`). Once a narrower range is identified (e.g., higher performance at batch sizes 16 or 32), switch to a continuous search to refine tuning:

   Update the sweep configuration:
   ```yaml
   batch_size:
     min: 16
     max: 32
   ```
### Monitoring and Interpreting Results
(/output/tuning_output_1.png)

#### Observing the Plots
The provided graph displays several key metrics, including `train_jaccard`, `train_loss`, `test_jaccard`, `test_loss`, `batch`, and `epoch`, plotted against the step count. These plots provide critical insights into model performance during hyperparameter tuning.

#### Why the Plots Show Sharp Up-and-Down Cycles
- The sharp oscillations in the plots indicate **trial-level evaluations**. Each upward and downward movement corresponds to **one trial within an experiment**.
- A single **experiment** consists of multiple trials, and the metrics are reset or updated at the beginning of each trial. For example, metrics like `jaccard` and `loss` reset or vary significantly as new hyperparameter combinations are evaluated.

#### Monitoring Final Metrics with Parallel Coordinate Plot
The parallel coordinate plot below visualizes the relationship between hyperparameters and the final metric `average_test_jaccard_index` across multiple trials in one experiment:

(/output/tuning_output_2.png)

- Each line in the plot represents run(experiment).
- The color gradient indicates the `average_test_jaccard_index` for each run, with brighter lines showing higher values.