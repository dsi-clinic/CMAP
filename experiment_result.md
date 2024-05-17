## Literature Review and Report for augmentation

### Resources:
[Possible technique](https://github.com/kornia/kornia-examples/blob/master/data_augmentation.ipynb)  
[Kornia package](https://kornia.readthedocs.io/en/latest/augmentation.html)
[Pytorch package]( https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py)  

### reference articles
* [Image Augmentation for Satellite Images](https://arxiv.org/pdf/2207.14580.pdf)
* [DATA AUGMENTATION APPROACHES FOR SATELLITE IMAGE SUPER-RESOLUTION](https://isprs-annals.copernicus.org/articles/IV-2-W7/47/2019/isprs-annals-IV-2-W7-47-2019.pdf)
*  [A review on remote sensing imagery augmentation using deep learning](https://www.sciencedirect.com/science/article/pii/S2214785322016820)

* Geometric Transformation
    * Horizontal / Vertical flip: highest accuracy
    * Zooming or scaling
    * Rotation
    * Limitation: geometric transformations like rotation, zooming and translation have limited use for medium and low-resolution satellite data as they do not provide enough variability
* Multi-temporal: image of the same place but different time
* Color-jittering: pixel values multiplied by different random numbers
* Edge detection - RGB to greyscale
* Contrast enhancement - histogram equalisation
* Unsharp masking

### Benchmark -- Accuracy with different experiment
#### Backbone and Weights

- batch_size = 16
- patch_size = 512
- lr = 1e-5
- epochs = 10
- patience = 5
- threshold = 0.01
- **num_trials = 10**
- spatial_aug_mode = "all"
- color_aug_mode = "all"

| Model | Encoder / Backbone | Weights | Final jaccard - mean | Final jaccard - std dev |
| :-------- | :----- | :------- | ---: | ---: |
|deeplabv3+ |resnet18| imagenet  | 0.563 | 0.055 |
|deeplabv3+ |resnet18| ssl       | 0.554 | 0.052 |
|deeplabv3+ |resnet18| swsl      | 0.538 | 0.057 |
|deeplabv3+ |resnet50| imagenet  | 0.571 | 0.041 |
|deeplabv3+ |resnet50| ssl       | 0.575 | 0.038 |
|deeplabv3+ |resnet50| swsl      | 0.546 | 0.058 |
|deeplabv3+ |resnet101| imagenet | **0.587** | 0.048 |
|deeplabv3+ |resnet152| imagenet | 0.560 | 0.057 |
|unet |resnet18| imagenet  | 0.561 | 0.039 |
|unet |resnet18| ssl       | 0.542 | 0.052 |
|unet |resnet18| swsl      | 0.524 | 0.058 |
|unet |resnet50| imagenet  | 0.555 | 0.044 |
|unet |resnet50| ssl       | 0.546 | 0.056 |
|unet |resnet50| swsl      | 0.575 | 0.031 |
|unet |resnet101| imagenet | 0.555 | 0.059 |
|unet |resnet152| imagenet | 0.551 | 0.081 |

#### Notes:
- based on code up to the commit [`76ed0b9`](https://github.com/dsi-clinic/2024-winter-cmap/commit/76ed0b93d09405e5da2f7f46c0b409e482ea167d)
- results stored in: /net/projects/cmap/workspaces/mingyan/model-perf

#### Augmentations
| Augmentation      | Indice | Training Jaccard Mean | Training Standard Dev | Test Jaccard Mean  | Test Standard Mean |
| -----------       | -----------    |  :----:  |  :----:  |  :----:  |  :----:  |
| Only Color        | [0,1,2,5]      |  0.726 |  0.007 | 0.545| 0.043|
| Only Gaussian     | [2,3]          |0.739   |0.006|0.594|0.038|
|Plasma Brightness + Gaussian|[2,3,4]|0.673   | 0.018| 0.579|0.008|
|Random Color + Gaussian |[0,1,2,3,5]| 0.696| 0.004| 0.575|0.052|
|Current |[0,1,2,3,5] |0.685|0.012|0.549|0.012|
|Without Plasma | [0,1,2,3,5,6,7] | 0.692|0.030|0.555|0.015|
|All|[0,1,2,3,4,5,6,7]| 0.614|0.020|0.600|0.067|


