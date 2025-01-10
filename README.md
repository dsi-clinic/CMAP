# CMAP

## Project Background
A comprehensive inventory of stormwater storage and green infrastructure (GI) assets is lacking across northeastern Illinois. Understanding the location of these assets is crucial for ensuring proper maintenance and gaining insights into potential impacts on water quality and stormwater management. An inventory could assist county and municipal stormwater engineers, public works officials, and others in ensuring proper maintenance and inform the development of watershed-based plans and resilience plans.

The Chicago Metropolitan Agency for Planning (CMAP) aims to utilize deep learning to map and identify locations of stormwater storage and related geographic features throughout Chicago and the surrounding area. To initiate the project, CMAP has provided labeled geographic features in Kane County, Illinois (provided by Kane County), to create a predictive deep learning model. This repository contains code to achieve the following objectives:

1. Obtain images corresponding to geographic features across Kane County.
2. Train and test various predictive deep learning models on surrounding geographies.
3. Apply Kane County data to identify stormwater basins in other Illinois counties.

## Project Goals

Several tasks are associated with this project:

1. Improve climate resiliency in northeastern Illinois by utilizing deep learning to map stormwater and green infrastructure from aerial data.
2. Develop deep learning models for aerial imaging data, focusing on green infrastructure and stormwater areas.
3. Train a model to identify different types of locations (e.g., wet ponds, dry-turf bottom, dry-mesic prairie, and constructed wetland detention basins) and then use this model to identify other areas of the region with these attributes.

These goals will be accomplished within the following pipeline structure:
1. Obtain corresponding NAIP images (retrieve_images.py and utils/get_naip_images.py).
2. Utilize a training loop (train.py) with configurations (configs/config.py) assigned to the cluster (.job), utilizing the model defined in utils/model.py and the custom Raster Dataset defined in utils/kc.py.

## Usage


### Environment Set Up

Before running the repository (see details below), you need to perform the following steps:
1. Install make if you have not already done so.
2. Ensure you have access to [Slurm] (python train.py configs.config --experiment_name baseline_v1).
3. Create and initiate a cmap specific mamba environment using the following steps:
    1) Install micromamba:
    ```
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
    ./bin/micromamba shell init -s bash -r ~/micromamba
    source ~/.bashrc
    micromamba config append channels conda-forge
    ```
    2) Create environment:
    ```
    micromamba create -y --name cmap python=3.10
    micromamba activate cmap
    micromamba install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
    git clone https://github.com/dsi-clinic/CMAP.git
    cd CMAP
    pip install -r requirements.txt
    ```
### Example of Training in Command Line
Next, you can train the model in an interactive session.
```
srun -p general --pty --cpus-per-task=8 --gres=gpu:1 --mem=128GB -t 0-06:00 /bin/bash

conda activate cmap

cd /home/YOUR_USERNAME/CMAP

python train.py configs.config --experiment_name <ExperimentName> --aug_type <aug> --split <split> --num_trial <num_trial>
```

Replace the <ExperimentName> with what you want to title the experiment. For example, if you want to title it baseline_v1, the code will be:
```python train.py configs.config --experiment_name baseline_v1```
.Aug_type, split, and num_trial are optional so you can ignore them if you don't have their parameters.

To speed up the development process and troubleshoot issues, you can use debug mode. This mode limits the training to a single epoch and skips subsequent epochs, allowing for faster code validation. To enable debug mode, use the --debug flag in the command:
```
python train.py configs.config --debug
```

### Example of Training with Slurm

If you have access to Slurm, you can also train model with it. For more information about how to use Slurm, please look at the information [here](https://github.com/uchicago-dsi/core-facility-docs/blob/main/slurm.md).

This option is best if you know that your code runs and you don't need to test anything with it.

To run this repo on the Slurm cluster after setting up your conda environment,

1. Go into the file called 'submit.sh'.
2. Change YOUR-USERNAME to your username.
3. To run the file on terminal, type: `sbatch submit.sh`. You can monitor whether your job is running with `squeue`.

Or, to run in an interactive session:
```
srun -p general --pty --cpus-per-task=8 --gres=gpu:1 --mem=128GB -t 0-06:00 /bin/bash

conda activate cmap

cd /home/YOUR_USERNAME/CMAP

python train.py configs.config --experiment_name <ExperimentName> --aug_type <aug> --split <split> --num_trial <num_trial>
```


## RiverDataset Information

To include RiverDataset in training:
* Set config.USE_RIVER_DATASET to True
* This will include BOTH KaneCounty and RiverDataset in train.py

*When USE_RIVER_DATASET is set to True, train.py becomes stuck after correctly loading the KC and RD datasets. **This needs to be debugged.***


## Git Usage

Before pushing changes to git, ensure that you're running `pre-commit run --all` to check your code against the linter.

## Repository Structure
### main repository
* **train.py**: containing code for training models
* **model.py**: defining the model framework used in training
* **experiment_result.md**: containing literauture review and experiments with differennt augmentation, backbone, and weights
* **sweep.job**: script used to run tuning with Wandb
* **requirements.txt**: containing required packages' information

### configs
containing config information
* **config.py**: default config for model training
* **sweep_config.yml**: config used for wandb sweep

### utils

Project python code. Contains various utility functions and scripts which support the main functionalities of the project and are designed to be reusable.
* **get_naip_images.py**
* **img_params.py** calculating images stats
* **plot.py** plotting image with labels
* **transform.py** Creating augmentation pipeline

### notebooks

Contains short, clean notebooks to demonstrate analysis. Documentation and descriptions included in the [README](notebooks/README.md) file.

### output

Contains example model output images and W&B tuning output plots.

### data

Contains details of acquiring all raw data used in repository. If data is small (<50MB) then it is okay to save it to the repo, making sure to clearly document how to the data is obtained.

If the data is larger than 50MB than you should not add it to the repo and instead document how to get the data in the README.md file in the data directory.

Source attribution and descriptions included in the [README](data/README.md) file.

## Final Results
The below results were obtained with these specifications:
* Classes: "POND" "WETLAND" "DRY BOTTOM - TURF" "DRY BOTTOM - MESIC PRAIRIE"
* Batch size: 16
* Patch size: 512
* Learning rate: 1E-5
* Number of workers: 8
* Epochs: 30 (maximum; early termination feature has been turned on)
* Augmentation: Random Contrast, Random Brightness, Gaussian Blur, Gaussian Noise, Random Satuation
* Number of trails: 5

Test Jaccard: mean: 0.589, standard deviation:0.075
Please refer to [experiment_report.md](https://github.com/dsi-clinic/2024-winter-cmap/blob/cleaning_code/experiment_result.md) for more experiments results

### example outputs
The model can detect ponds fairly accurately:
![output_image1](/output/example_images/DL_ResNet50_imagenet_v1/epoch-14/test_sample-14.0.0.png)
![output_image2](/output/example_images/DL_ResNet50_imagenet_v1/epoch-14/test_sample-14.0.7.png)
![output_image3](/output/example_images/DL_ResNet50_imagenet_v1/epoch-14/test_sample-14.0.8.png)

There needs to be some tweaks for the model to better identify wetlands and dry bottom turf stormwater infrastructure:
![output_image4](/output/example_images/DL_ResNet50_imagenet_v1/epoch-14/test_sample-14.0.2.png)
![output_image5](/output/example_images/DL_ResNet50_imagenet_v1/epoch-14/test_sample-14.0.5.png)

There also needs to be adjustments made to the model to account for false positives:
![output_image6](/output/example_images/DL_ResNet50_imagenet_v1/epoch-14/test_sample-14.1.6.png)
![output_image7](/output/example_images/DL_ResNet50_imagenet_v1/epoch-14/test_sample-14.1.10.png)

### Running Hyperparameter Sweep with SLURM and Weights & Biases
This guide provides step-by-step instructions to set up and run a hyperparameter sweep using **SLURM** and **Weights & Biases (W&B)** in a micromamba environment.
#### Prerequisites

1. **Create a Weights & Biases Account**  
   Sign up for a free W&B account at [https://wandb.ai](https://wandb.ai) if you don’t have one already.

2. **Install Micromamba**  
   Ensure micromamba is installed, and your environment is set up with all dependencies.

3. **Install Weights & Biases (wandb)**  
   Install W&B in your micromamba environment by running:

   ```
   micromamba install -c conda-forge wandb
   ```
4. **Log in to W&B**
    Log into your W&B account by running:
   ```
   wandb login
   ```

#### Initialize the Sweep in W&B:
Run the following command to create a sweep ID in W&B:
```
wandb sweep /home/<YOUR-USERNAME>/CMAP/configs/sweep_config.yml
```
#### View Sweep in W&B:
After running wandb sweep command above, a sweep ID will be generated and you can click the link shown in terminal to view the sweep in W&B interface.

#### Running a Sweep Directly from the Terminal
If you prefer to run a hyperparameter sweep directly from the terminal instead of submitting it as a SLURM job, use the following command:

```
wandb agent <W&B_organization_name>/<project_name>/<sweep_ID> --count <number_of_experiments>
```
Example:
```
wandb agent dsi-clinic-cmap/CMAP/6tmqjcu4 --count 5
```
In this example:
dsi-clinic-cmap is the organization name in W&B.
CMAP is the project name.
6tmqjcu4 is the sweep ID.
--count 5 specifies that the agent should conduct 5 runs(experiments).
This is particularly useful for quick tests or for running sweeps locally without using a job scheduling system like SLURM.

## Running a Sweep using a SLURM Job Script
1. **Edit the `sweep.job` File**:
   - Replace `<YOUR-USERNAME>` with your username.
   - Replace `<W&B_organization_name>` and `<sweep_id>` with the appropriate values from your W&B project.

2. **Submit the Job and Monitor Its Status**:
   ```bash
   sbatch sweep.job
   squeue -u <YOUR-USERNAME>
   ```
Please refer to [future-work documentation](future-work.md) for more detailed information.

## Collaborators
- Matthew Rubenstein - rubensteinm@uchicago.edu
- Spencer Ellis - sjne@uchicago.edu
- Tamami Tamura - tamamitamura@uchicago.edu
- Mingyan Wang - mingyan@uchicago.edu
- Miao Li - mli628@uchicago.edu
- Grey Xu - greyxu@uchicago.edu

## Collaborators- Fall 2024
- Anna Poon - annapoon@uchicago.edu
- Collin Kim - collinkim415@uchicago.edu
- Mohit Kumar Sathishkumar - mohitk@uchicago.edu
- Xiaoyue Wei - xiaoyue1@uchicago.edu
- Cassandra Haas - cassandrahaas@uchicago.edu
