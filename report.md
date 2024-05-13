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
* only color augmentations [0,1,2,5]: 
   * Training: average: 0.726, standard deviation: 0.007
   * Test: mean: 0.545, standard deviation:0.043
* only gaussian augmentations [2,3]:
   * Training: average: 0.739, standard deviation: 0.006
   * Test: mean: 0.594, standard deviation:0.038
* Plasma Brightness + gaussian: [2,3,4]
   * Training: average: 0.673, standard deviation: 0.018
   * Test: mean: 0.579, standard deviation:0.008
* random color + gaussian [0,1,2,3,5] [running]
