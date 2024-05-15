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

### Benchmark -- Accuracy with different augmentation methods
| Augmentation      | Indice | Training Jaccard Mean | Training Standard Dev | Test Jaccard Mean  | Test Standard Mean |
| -----------       | -----------    |  :----:  |  :----:  |  :----:  |  :----:  |
| Only Color        | [0,1,2,5]      |  0.726 |  0.007 | 0.545| 0.043|
| Only Gaussian     | [2,3]          |0.739   |0.006|0.594|0.038|
|Plasma Brightness + Gaussian|[2,3,4]|0.673   | 0.018| 0.579|0.008|
|Random Color + Gaussian |[0,1,2,3,5]| 0.696| 0.004| 0.575|0.052|
|All|[0,1,2,3,4,5,6]| 0.614|0.020|0.600|0.067|


