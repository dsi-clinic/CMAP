import os

import segmentation_models_pytorch as smp
import timm
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import NAIP, stack_samples
from torchgeo.models import ResNet18_Weights
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler

# from config import dsi
from kane_county import KaneCounty

# The below directory definitions are in place until the config/dsi.py file
# is merged into the main branch, as shown in the above commented import
DATA_DIR = "/net/projects/cmap/data"
KC_SHAPE_DIR = os.path.join(DATA_DIR, "kane-county-data")
KC_IMAGE_DIR = os.path.join(DATA_DIR, "KC-images")
KC_MASK_DIR = os.path.join(DATA_DIR, "KC-masks/top-structures-masks")
# -----------

# Imports correct image directories
naip = NAIP(KC_IMAGE_DIR)
kc = KaneCounty(KC_MASK_DIR)
dataset = naip & kc

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Pre-trained weights
weights = ResNet18_Weights.SENTINEL2_ALL_MOCO

# Initiates model
in_chans = weights.meta["in_chans"]
model = timm.create_model("resnet18", in_chans=in_chans, num_classes=10)
model = model.load_state_dict(
    weights.get_state_dict(progress=True), strict=False
)

# Unless the dataset has a 'num_classes' attribute that specifies the number of classes
num_classes = 5  # dataset.num_classes


# Define the model architecture for segmentation
model = smp.Unet(
    encoder_name="resnet18",  # Encoder backbone
    in_channels=in_chans,
    classes=num_classes,  # Number of output classes
)

# Alternatively, you can choose from other models in segmentation_models_pytorch
# smp.DeepLabV3(encoder_name="resnet18",in_channels=in_chans,classes=num_classes)

# Sampler of size 1000 (dimensions of patch), length 10 (number of samples)
train_sampler = RandomGeoSampler(dataset, size=1000, length=10)

# Combines dataset and sampler to iterate over data
train_dataloader = DataLoader(
    dataset, sampler=train_sampler, collate_fn=stack_samples
)

# Sampler of size 1000 (dimensions of patch), length 10 (number of samples)
stride = max(len(dataset) // 10, 1)
test_sampler = GridGeoSampler(dataset, size=1000, stride=stride)

# Combines dataset and sampler to iterate over data
test_dataloader = DataLoader(
    dataset, sampler=train_sampler, collate_fn=stack_samples
)

loss_fn = (
    smp.utils.losses.DiceLoss()
)  # nn.CrossEntropyLoss() # Change loss later
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# Training Loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# Saving model parameters to an internal state dictionary
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
