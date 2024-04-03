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
* With plasma: epoch 15:  Jaccard index: 0.535157, Avg loss: 0.209104
* With default: Jaccard index: 0.630060, Avg loss: 0.227179 
* With all: 9 epoch: Jaccard index: 0.498313, Avg loss: 0.204150 (so far the higheset)
* with box blur: 10 epoch: Jaccard index: 0.528737, Avg loss: 0.227576
* with color jitter: 11 epoch: Jaccard index: 0.594752, Avg loss: 0.251863 
