# Model performance 

with configs:
- batch_size = 16
- patch_size = 512
- lr = 1e-5
- num_workers = 8
- epochs = 30
- loss_function = "jaccardloss"
- patience = 5
- threshold = 0.01
- num_trials = 3

based on code up to the commit [`db972b0`](https://github.com/dsi-clinic/2024-winter-cmap/commit/db972b0309f2f38ff498dbd9120fd27257840f30)


| Model | Encoder / Backbone | Weights | Final loss (avg) | Final jaccard (avg) |
| :-------- | :----- | :------- | ---: | ---: |
|deeplabv3+ |resnet18| imagenet  | 0.4098 | 0.5810 |
|deeplabv3+ |resnet18| ssl       | 0.4289 | 0.5379 |
|deeplabv3+ |resnet18| swsl      | 0.4143 | 0.5708 |
|deeplabv3+ |resnet50| imagenet  | 0.4276 | 0.5488 |
|deeplabv3+ |resnet50| ssl       | 0.4171 | 0.5788 |
|deeplabv3+ |resnet50| swsl      | 0.3918 | **0.6116** |
|deeplabv3+ |resnet101| imagenet | 0.3689 | 0.5958 |
|deeplabv3+ |resnet152| imagenet | 0.4039 | 0.5646 |
|unet |resnet18| imagenet  | 0.4412 | 0.5333 |
|unet |resnet18| ssl       | 0.4276 | 0.5580 |
|unet |resnet18| swsl      | 0.4397 | 0.5575 |
|unet |resnet50| imagenet  | 0.3844 | **0.6112** |
|unet |resnet50| ssl       | 0.4302 | **0.6045** |
|unet |resnet50| swsl      | 0.3902 | 0.5808 |
|unet |resnet101| imagenet | 0.4506 | 0.5470 |
|unet |resnet152| imagenet | 0.3780 | **0.6048** |