"""
Use our data to create a folder with .png images to train diffusion models on. 

"""
from PIL import Image
import numpy as np
import torch
from data.sampler import BalancedGridGeoSampler, BalancedRandomBatchGeoSampler
import random
import logging
import sys
import torch
import os
from torchgeo.datasets import NAIP, random_bbox_assignment, stack_samples
from torch.utils.data import DataLoader, TensorDataset

naip_dataset = NAIP(os.path.join("/net/projects/cmap/data", "KC-images"))
# make it import from config

seed = random.randint(0, sys.maxsize)
logging.info("Dataset random split seed: %d", seed)
generator = torch.Generator().manual_seed(seed)

train_sampler = BalancedRandomBatchGeoSampler(
    config={
        "dataset": naip_dataset,
        "size": 256,
        "batch_size": len(naip_dataset),
    }
)

train_dataloader = DataLoader(
    dataset=naip_dataset,
    batch_sampler=train_sampler,
    collate_fn=stack_samples,
    num_workers=8,
)


def save_naip_images(dataloader, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        # Assuming the image data is in 'image' key
        images = batch["image"]

        for i, image in enumerate(images):
            # Convert to numpy and ensure it's in (C, H, W) format
            if isinstance(image, torch.Tensor):
                image = image.numpy()

            # Ensure image is in (C, H, W) format
            if image.shape[0] not in [1, 3, 4]:  # If first dimension is not channel
                image = np.transpose(image, (2, 0, 1))  # Transpose to (C, H, W)

            # Transpose to (H, W, C) for saving
            image_np = np.transpose(image, (1, 2, 0))

            # Normalize to 0-255 range and convert to uint8
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
            image_np = (image_np * 255).astype(np.uint8)

            # Create PIL Image
            pil_image = Image.fromarray(image_np)

            # Save the image
            image_name = f"naip_image_batch{batch_idx:05d}_sample{i:05d}.png"
            pil_image.save(os.path.join(output_dir, image_name))

        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx+1}/{total_batches} batches")


# Use your NAIP dataset

if __name__ == "__main__":
    # Create a DataLoader
    batch_size = 1  # Adjust as needed

    # Save images to a new directory
    save_naip_images(train_dataloader, "/net/scratch/ijain1/naip_dataset")
    # make the file use config as well
