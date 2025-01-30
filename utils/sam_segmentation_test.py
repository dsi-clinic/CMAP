from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from pycocotools import mask as mask_utils
import random

# Paths
model_checkpoint = "/home/gregoryc25/CMAP/segment-anything/sam_vit_h.pth"
images_dir = "/home/gregoryc25/CMAP/segment-anything/demo/src/assets/data"
output_dir = "/home/gregoryc25/CMAP/segment-anything/demo/src/assets/data/outputs"

# Function to generate random color
def random_color():
    return [random.random(), random.random(), random.random()]

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load SAM model
sam = sam_model_registry["vit_h"](checkpoint=model_checkpoint)
sam.to(device="cuda")

# Initialize predictor
predictor = SamPredictor(sam)

# Process images
for img_file in os.listdir(images_dir):
    if img_file.endswith((".jpg", ".png")):
        img_path = os.path.join(images_dir, img_file)
        json_path = os.path.join(images_dir, img_file.replace(".jpg", ".json").replace(".png", ".json"))

        # Check if JSON file exists
        if not os.path.exists(json_path):
            print(f"No JSON found for {img_file}. Skipping...")
            continue

        # Load the image
        image = np.array(Image.open(img_path).convert("RGB"))
        predictor.set_image(image)

        # Load annotations from JSON
        with open(json_path, "r") as f:
            annotations_data = json.load(f)

        annotations = annotations_data.get("annotations", [])
        if not annotations:
            print(f"No annotations found for {json_path}. Skipping...")
            continue

        print(f"Processing {img_file}...")

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)

        # Process each annotation
        for i, annotation in enumerate(annotations):
            # Decode segmentation mask from COCO RLE format
            rle = annotation["segmentation"]
            mask = mask_utils.decode(rle).astype(np.uint8)

            # Extract polygonal shapes from the mask
            contours = find_contours(mask, 0.5)

            # Assign a random color for each annotation
            color = random_color()

            # Draw each contour
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=2)

            # Annotate with a polygon ID
            if len(contours) > 0:
                x, y = contours[0][0]  # Use the first point of the first contour
                ax.text(
                    x, y, f"Segment {i+1}", color='white', fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
                )

        plt.axis("off")
        plt.title(f"Polygon Segmentation for {img_file}")

        # Save visualization
        vis_output_path = os.path.join(output_dir, f"{img_file}_polygon_overlay.png")
        plt.savefig(vis_output_path)
        plt.close()
        print(f"Saved visualization for {img_file} at {vis_output_path}")
