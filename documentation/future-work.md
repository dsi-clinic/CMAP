
# Future Work

## Hyperparameter Tuning
### How to submit a slurm job for tuning

1. **Edit the `sweep.job` File**:
   - Replace `<YOUR-USERNAME>` with your username.
   - Replace `<W&B_organization_name>` and `<sweep_id>` with the appropriate values from your W&B project.

2. **Submit the Job and Monitor Its Status**:
   ```bash
   sbatch sweep.job
   squeue -u <YOUR-USERNAME>
   ```

### Run Sweeps Using Multiple Agents 
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
   
### Monitoring and Interpreting Results
![Experiment metrics plots](/output/tuning_plots/tuning_output_1.png)

#### Observing the Plots
The provided graph displays several key metrics, including `train_jaccard`, `train_loss`, `test_jaccard`, `test_loss`, `batch`, and `epoch`, plotted against the step count. These plots provide critical insights into model performance during hyperparameter tuning.

#### Why the Plots Show Sharp Up-and-Down Cycles
- The sharp oscillations in the plots indicate **trial-level evaluations**. Each upward and downward movement corresponds to **one trial within an experiment**.
- A single **experiment** consists of multiple trials, and the metrics are reset or updated at the beginning of each trial. For example, metrics like `jaccard` and `loss` reset or vary significantly as new hyperparameter combinations are evaluated.

### Initial Experiments Results Using Bayesian Search
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
- **Best Result**: average_test_jaccard **0.5585** 
- Insights: Lower patch_size, batch_size, and color_brightness yielded better results.

The parallel coordinate plot below visualizes the relationship between hyperparameters and the final metric `average_test_jaccard_index` across multiple trials in one experiment:

![Correlation Plot](/output/tuning_plots/correlation_plot_6params.png)
![Parallel Coordinate Plot](/output/tuning_plots/tuning__plot_6params.png)
- Each line in the plot represents run(experiment).
- The color gradient indicates the `average_test_jaccard_index`   for each run, with brighter lines showing higher values.

- **Case with 5 Parameters**: eliminating gaussian_blur_sigma and gaussian_blur_kernel and add dropout
  Parameters:
  ```yaml
  batch_size:
    values: [32, 64]
  learning_rate:
    values: [0.0005, 0.001]
  patch_size:
    values: [256, 512, 768]
  color_brightness:
    values: [0, 0.3, 0.5]
  dropout:
    values: [0, 0.15, 0.3]
  ```
- **Best Result**: average_test_jaccard **0.5944**, with this following config: 
```yaml
  batch_size:
    values: 32
  learning_rate:
    values: 0.0005
  patch_size:
    values: 256
  color_brightness:
    values: 0.5
  dropout:
    values: 0
  ```
-Insights: Higher importance for dropout, patch_size, and color_brightness. Lower patch_size correlated with better results.
![Correlation Plot](/output/tuning_plots/correlation_plot_5params.png)
![Parallel Coordinate Plot](/output/tuning_plots/tuning_plot_5params.png)

- **Case with 4 Parameters and continous search**:
```yaml
  batch_size:
    values: [32, 64]
  learning_rate:
    distribution: loguniform
    min: 0.0001
    max: 0.005
  patch_size:
    values: 256
  color_brightness:
    distribution: uniform
    min: 0
    max: 0.5
  dropout:
    distribution: uniform
    min: 0
    max: 0.3
  ```
  ![Correlation Plot](/output/tuning_plots/correlation_plot_4params.png)
  ![Parallel Coordinate Plot](/output/tuning_plots/tuning_plot_4params.png)
  - **Best Result**: average_test_jaccard **0.583**
  Based on the results above, the next step is to further narrow down
 the search range. For example, the next step can be fix batch size to 32, narrow down the search range for learning_rate, color_brightne and dropout.
 Here's a sample sweep configuration for next experiment. 
  ```yaml
  batch_size:
    values: 32
  learning_rate:
    distribution: uniform
    min: 0.0001
    max: 0.0005
  patch_size:
    values: 256
  color_brightness:
    distribution: uniform
    min: 0.05
    max: 0.25
  dropout:
    distribution: uniform
    min: 0
    max: 0.2
  ```

## Class Imbalance: Woring in Progress PR #147
### Class distribution by freqency and area size 
| BasinType                  | count | total_area    | 
|----------------------------|-------|---------------|
| POND                       | 2375  | 1.702098e+07  |
| WETLAND                    | 1655  | 9.540348e+06  |
| DRY BOTTOM - TURF          | 936   | 2.831817e+06  |
| DRY BOTTOM - MESIC PRAIRIE | 223   | 1.074185e+06  | 

![Frequency of each class:](/documentation/frequency_by_class_plot.png)
![Area size of each class:](/documentation/area_by_class_plot.png)

### Next Steps:
Due to the underrepresentation of DRY BOTTOM - TURF and DRY BOTTOM - MESIC PRAIRIE, we will explore oversampling these classes during data preparation. We have implemented a test sampler based on the RandomBatchGeoSampler from the TorchGeo package. The next step is to refine this sampler to weight classes (POND, WETLAND, TURF, and MESIC PRAIRIE) according to both their frequency and total area, thereby mitigating class imbalance for improved model performance.
