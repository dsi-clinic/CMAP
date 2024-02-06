## imports
import torch
import torchvision
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from torchgeo.datamodules import InriaAerialImageLabelingDataModule
from torchgeo.datasets import CDL, Landsat7, Landsat8, VHR10, stack_samples
from torchgeo.samplers import RandomGeoSampler
from torchgeo.trainers import SemanticSegmentationTask
import matplotlib.pyplot as plt
import timm
from torchgeo.models import ResNet18_Weights
import numpy as np
import kornia as K


## dataset
# KC data class defined in model/torchgeo branch: data/kane_county.py

# For reference, directories defined below:
"""
DATA_DIR = "/net/projects/cmap/data"
KC_SHAPE_DIR = os.path.join(DATA_DIR, "kane-county-data")
KC_IMAGE_DIR = os.path.join(DATA_DIR, "KC-images")
KC_MASK_DIR = os.path.join(DATA_DIR, "KC-masks/top-structures-masks")
"""
# KaneCounty class defined by Spencer's code
KC_images_data = KaneCounty(root=KC_IMAGE_DIR) # check if we need to define this root, or just leave it as root
KC_masks_data = KaneCounty(root=KC_MASK_DIR)
img_masks_data = KC_images_data & KC_masks_data

## Split the dataset with the combined images and masks
indices = torch.randperm(len(img_masks_data)).tolist()
dataset = torch.utils.data.Subset(img_masks_data, indices[:-50])
dataset_test = torch.utils.data.Subset(img_masks_data, indices[-50:])

## Create TorchGeo GeoDataset sampler
patch_dim = 256
num_patches_per_epoch = 10000
sampler = RandomGeoSampler(dataset, size=patch_dim, length=num_patches_per_epoch)

patch_dim_test = 256
num_patches_per_epoch_test = 100
sampler_test = RandomGeoSampler(dataset_test, size=patch_dim_test, length=num_patches_per_epoch_test)

## Combine dataset and sampler into a single DataLoader
batch_size = 30
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=stack_samples,
                        shuffle=True)

batch_size_test= 10
dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test, sampler=sampler_test, collate_fn=stack_samples,
                        shuffle=True)

## Define model specifics
# Model
weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
model = timm.create_model("resnet18", in_chans=weights.meta["in_chans"], num_classes=10)
model = model.load_state_dict(weights.get_state_dict(progress=True), strict=False)

# Loss function
loss_fn = torch.nn.CrossEntropyLoss() 
# requires 1 channel images, but masks have 4 channels; maybe loss fn expects one-hot-encoded output?
#may need to tweak model definition
# masks are not one-hot-encoded

# Optimizer = stochastic gradient descent with momentum
"""
Learning rate - size of steps the optimizer takes
Momentum - nudges optimizer in direction of strongest gradient over multiple steps
Other optimization algos: Adagrad, Adam, averaged SGD
"""
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train one epoch
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for batch in dataloader:
        # Extract the images and masks
        image = batch["image"]
        target = batch["mask"]
        
        ## Transforms
        transforms = AugmentationSequential( # Check function
            MinMaxNormalize(mins, maxs),
            # Check augmentations to see if they produce results we'd actually see
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomPlasmaShadow(roughness=(0.1, 0.7), shade_intensity=(-1.0, 0.0), shade_quantity=(0.0, 1.0), keepdim=True),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25),
            K.RandomResizedCrop(size=(patch_dim, patch_dim), scale=(0.08, 1.0), p=0.25),
            data_keys=["image"],
        ).to("cuda") # Kornia runs transforms on GPU, whilst most libraries run on CPU
        images = transforms(images)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        # Check if the model can take the image inputs like this
        outputs = model(image)

        # Compute the loss and its gradients
        # Check if the loss_fn can calculate loss with img input parameters
        loss = loss_fn(outputs, target)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        # check with implementations of loss, IoU
        if i % 1000 == 999: #check i 
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Calculate IoU across a batch - trying to optimize IoU, ideally IoU goes up and loss goes down. At some point these two start to plateau. 
# Frequently, loss will plateau before the IoU, but we stop training when IoU plateaus
SMOOTH = 1e-6
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded # Value for each batch

# check datetime and SummaryWriter what they depend on
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

# Total number of epochs
num_epochs = 15

best_vloss = 1_000_000.

# Arrays storing losses for plotting
train_losses = np.array()
val_losses = np.array()
val_ious = np.array()

## Training Loop
for epoch in range(num_epochs):

    print('EPOCH {}:'.format(epoch_number + 1))

    ## Train
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)
    train_losses = np.append(train_losses, avg_loss)

    ## Evaluate
    running_vloss = 0.0
    running_viou = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    vbatch_number = 0
    with torch.no_grad():
        for vbatch in enumerate(dataloader_test):
            vimage = vbatch["image"]
            vtarget = vbatch["mask"]
            voutputs = model(vimage)
            vloss = loss_fn(voutputs, vtarget)
            running_vloss += vloss
            vbatch_iou = iou_pytorch(voutputs, vtarget)
            running_viou += vbatch_iou
            vbatch_number += 1

    avg_vloss = running_vloss / vbatch_number
    avg_viou = running_viou / vbatch_number
    val_losses = np.append(val_losses, avg_vloss)
    val_ious = np.append(val_ious, avg_viou)

    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    print('IOU valid {}'.format(avg_viou))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss and IoU',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss, 'Validation IoU' : avg_viou },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

## Plotting
plt.plot(train_losses, 'r')
plt.plot(val_losses, color='green')
plt.plot(val_ious, color='blue', linestyle='dashed')
plt.title("Validation IoU, Training Loss, and Validation Loss per Epoch")
plt.xlabel("Epoch")
plt.legend(['Training Loss', 'Validation Loss', 'Validation IoU'])

plt.show()

## Notes
"""
- when running inference, use GridSampler
- consider TorchGeo's SemanticSegmentationTask 
- Look at segmentation models: https://github.com/satellite-image-deep-learning/techniques#Segmentation
"""
