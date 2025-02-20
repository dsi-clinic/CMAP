import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassJaccardIndex
from pathlib import Path
from data.sampler import BalancedRandomBatchGeoSampler
from configs import config
from torch.utils.data import DataLoader
from torchgeo.datasets import NAIP, stack_samples
from data.kc import KaneCounty
from segment_anything import SamPredictor, sam_model_registry
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Segmentation Script')
parser.add_argument('--use_all_classes', action='store_true', help='Use all available classes instead of default config')
parser.add_argument('--max_objects', type=int, default=None, help='Maximum number of objects to process')
args = parser.parse_args()

OUTPUT_DIR = "kane_segmentation_results"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

naip_dataset = NAIP("/net/projects/cmap/data/KC-images")
shape_path = Path(config.KC_SHAPE_ROOT) / config.KC_SHAPE_FILENAME

# Define class labels
if args.use_all_classes:
    all_labels = {
        0: "BACKGROUND",
        1: "POND",
        2: "WETLAND",
        3: "DRY BOTTOM - TURF",
        4: "DRY BOTTOM - MESIC PRAIRIE",
        5: "DEPRESSIONAL STORAGE",
        6: "DRY BOTTOM - WOODED",
        7: "POND - EXTENDED DRY",
        8: "PICP PARKING LOT",
        9: "DRY BOTTOM - GRAVEL",
        10: "UNDERGROUND",
        11: "UNDERGROUND VAULT",
        12: "PICP ALLEY",
        13: "INFILTRATION TRENCH",
        14: "BIORETENTION",
        15: "UNKNOWN",
    }
else:
    all_labels = config.KC_LABELS

# Configure dataset
dataset_config = (
    config.KC_LAYER,
    all_labels,
    config.PATCH_SIZE,
    naip_dataset.crs,
    naip_dataset.res,
)

kc_dataset = KaneCounty(shape_path, dataset_config)
train_dataset = naip_dataset & kc_dataset

train_sampler = BalancedRandomBatchGeoSampler(
    config={
        "dataset": train_dataset,
        "size": config.PATCH_SIZE,
        "batch_size": 1,
    }
)

plot_dataloader = DataLoader(
    dataset=train_dataset,
    batch_sampler=train_sampler,
    collate_fn=stack_samples,
    num_workers=config.NUM_WORKERS,
)

sam = sam_model_registry["vit_h"](checkpoint="/home/gregoryc25/CMAP/segment-anything/sam_vit_h.pth")
predictor = SamPredictor(sam)

MAX_SAVED_OUTPUTS = 20

iou_metric = MulticlassJaccardIndex(num_classes=len(all_labels), ignore_index=0, average="macro")

iou_results = {}

total_objects = 0
valid_objects = 0
skipped_objects = 0
saved_images = 0

for obj_id, sample in enumerate(plot_dataloader):
    if args.max_objects and total_objects >= args.max_objects:
        break
    total_objects += 1
    print(f"\nProcessing object {obj_id}...")

    img_tensor = sample["image"][0]
    gt_mask = sample["mask"][0].numpy()

    img_np = img_tensor[:3].cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0)).astype(np.uint8)

    unique_labels = np.unique(gt_mask)
    valid_labels = unique_labels[unique_labels > 0]

    if len(valid_labels) == 0:
        print(f"Skipping object {obj_id} - No valid segmentation regions found!")
        skipped_objects += 1
        continue

    inferred_class_label = np.random.choice(valid_labels)
    print(f"Inferred class label: {inferred_class_label} ({all_labels.get(inferred_class_label, 'UNKNOWN')})")

    valid_objects += 1

    predictor.set_image(img_np)

    ys, xs = torch.where(torch.tensor(gt_mask) == inferred_class_label)
    random_idx = torch.randint(0, len(xs), (1,)).item()
    seed_x, seed_y = xs[random_idx].item(), ys[random_idx].item()

    seed_coords = np.array([[seed_x, seed_y]])
    seed_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=seed_coords,
        point_labels=seed_label,
        multimask_output=False
    )

    pred_mask = masks[0]

    gt_tensor = torch.tensor(gt_mask, dtype=torch.int64).unsqueeze(0)
    pred_tensor = torch.tensor(pred_mask, dtype=torch.int64).unsqueeze(0)

    iou_score = iou_metric(pred_tensor, gt_tensor).item()

    if inferred_class_label not in iou_results:
        iou_results[inferred_class_label] = []
    iou_results[inferred_class_label].append(iou_score)

    if saved_images < MAX_SAVED_OUTPUTS:
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.scatter(seed_x, seed_y, color="red", s=20, edgecolors="black", linewidth=0.5)
        plt.title("Image (Seed Point)")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask, cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap="gray")
        plt.title(f"Predicted Mask (IoU: {iou_score:.2f})")
        plt.axis("off")

        save_path = f"{OUTPUT_DIR}/object_{obj_id}.png"
        try:
            plt.savefig(save_path)
            plt.close()
            saved_images += 1
            print(f"✅ Image saved successfully: {save_path}")
        except Exception as e:
            print(f"❌ Error saving image: {e}")

print("\nSegmentation Complete!")
