
import os
#import tempfile

from torch.utils.data import DataLoader

from torchgeo.datasets import NAIP, ChesapeakeDE, stack_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchgeo.models import ResNet18_Weights
import timm
import segmentation_models_pytorch as smp
from torchgeo.datasets import NAIP
from kane_county import KaneCounty

DATA_DIR = "/net/projects/cmap/data"
KC_SHAPE_DIR = os.path.join(DATA_DIR, "kane-county-data")
KC_IMAGE_DIR = os.path.join(DATA_DIR, "KC-images")
KC_MASK_DIR = os.path.join(DATA_DIR, "KC-masks/top-structures-masks")

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


weights = ResNet18_Weights.SENTINEL2_ALL_MOCO

in_chans = weights.meta["in_chans"]
model = timm.create_model("resnet18", in_chans=in_chans, num_classes=10)
model = model.load_state_dict(weights.get_state_dict(progress=True), strict=False)

sampler = RandomGeoSampler(dataset, size=1000, length=10)

dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)

loss_fn = smp.utils.losses.DiceLoss() #nn.CrossEntropyLoss() # Change loss later
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


#for sample in dataloader:
#    image = sample["image"]
#    target = sample["mask"]

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(dataloader, model, loss_fn, optimizer)
    #test(test_dataloader, model, loss_fn)
print("Done!")

# Use dice loss or jacard loss for segmentation
    #Ensure everything is enclosed in torch.nograd when evaluating, so not here, but next
    # step, to make sure it doesn't continue training
    #Transfer learning