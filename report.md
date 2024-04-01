## Literature Review and Report for augmentation

### Resources:
possible technique[https://github.com/kornia/kornia-examples/blob/master/data_augmentation.ipynb]
Kornia package[https://kornia.readthedocs.io/en/latest/augmentation.html]

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