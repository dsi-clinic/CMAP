# Model performance 

#### configs:
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
