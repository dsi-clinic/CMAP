#!/usr/bin/env python

"""Segmentation pipeline for processing NAIP imagery with Segment Anything (SAM)."""

import argparse
import csv
import os  # Fixed: Imported `os` to resolve undefined name error
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import NAIP, stack_samples

from configs import config
from data.kc import KaneCounty
from data.sampler import BalancedRandomBatchGeoSampler
from segment_anything_source_code import SamPredictor, sam_model_registry

# Use a non-interactive backend for matplotlib (headless mode)
matplotlib.use("Agg")

# Constant for channel check
NUM_CHANNELS = 3


def compute_instance_iou(pred_mask, gt_mask, instance_id):
    """Compute IoU for one predicted mask vs. one ground-truth label ID.

    Args:
        pred_mask (np.ndarray): Predicted segmentation mask.
        gt_mask (np.ndarray): Ground-truth segmentation mask.
        instance_id (int): The ground-truth label ID.

    Returns:
        float: IoU value.
    """
    gt_instance = (gt_mask == instance_id).astype(np.uint8)
    pred_instance = (pred_mask > 0).astype(np.uint8)  # 1=object, 0=bg
    intersection = np.logical_and(gt_instance, pred_instance).sum()
    union = np.logical_or(gt_instance, pred_instance).sum()
    return float(intersection) / union if union != 0 else 0.0


def main():
    """Processes NAIP imagery by running Segment Anything (SAM) on each patch.

    Optionally runs in parallel when using an HPC array job.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Limit to this many patches; -1 for all.",
    )
    parser.add_argument(
        "--subset_index",
        type=int,
        default=None,
        help="Subset index if running multiple jobs.",
    )
    parser.add_argument(
        "--total_subsets",
        type=int,
        default=None,
        help="Total subsets for parallel processing.",
    )
    args = parser.parse_args()

    # Output directory setup
    home_dir = Path.home()
    base_out_dir = home_dir / "CMAP" / "segment_anything_CMAP" / "kc_sam_outputs"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    job_id = (
        Path(f"kc_sam_run_{Path.home().name}")
        if "SLURM_JOB_ID" not in os.environ  # Fixed: Now `os` is properly imported
        else Path(
            f"kc_sam_run_{os.getenv('SLURM_JOB_ID')}"
        )  # Fixed: No longer undefined
    )
    out_dir = base_out_dir / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Using single output directory: {out_dir}")

    # Load dataset
    naip_dataset = NAIP("/net/projects/cmap/data/KC-images")
    shape_path = Path(config.KC_SHAPE_ROOT) / config.KC_SHAPE_FILENAME

    dataset_config = (
        config.KC_LAYER,
        config.KC_LABELS,
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

    total_samples = len(plot_dataloader) if plot_dataloader else None

    print("[INFO] Loading SAM model...")
    sam_checkpoint = (
        home_dir / "CMAP" / "segment_anything_source_code" / "sam_vit_h.pth"
    )
    sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
    predictor = SamPredictor(sam)

    per_class_ious = {}
    num_processed = num_skipped = num_total_iou_calcs = 0

    print("[INFO] Starting segmentation loop...")

    for sample_idx, batch in enumerate(plot_dataloader):
        if args.max_samples != -1 and sample_idx >= args.max_samples:
            break

        if (
            args.total_subsets
            and args.subset_index is not None
            and total_samples is not None
        ):
            subset_size = total_samples // args.total_subsets
            start_idx, end_idx = (
                args.subset_index * subset_size,
                (args.subset_index + 1) * subset_size,
            )
            if args.subset_index == args.total_subsets - 1:
                end_idx = total_samples
            if not (start_idx <= sample_idx < end_idx):
                continue

        print(
            f"Processing patch {sample_idx + 1}/{total_samples if total_samples else '?'}..."
        )

        img_tensor, mask_tensor = batch["image"][0], batch["mask"][0]
        if mask_tensor.dim() == NUM_CHANNELS and mask_tensor.shape[0] == 1:
            mask_tensor = mask_tensor.squeeze(0)

        valid_labels = torch.unique(mask_tensor)[torch.unique(mask_tensor) > 0]
        if len(valid_labels) == 0:
            num_skipped += 1
            continue

        predictor.set_image(
            img_tensor[:3].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        )
        gt_mask_np = mask_tensor.cpu().numpy()

        for label_id in valid_labels:
            ys, xs = torch.where(mask_tensor == label_id)
            if len(xs) == 0:
                continue

            seed_x, seed_y = (
                xs[torch.randint(0, len(xs), (1,))].item(),
                ys[torch.randint(0, len(ys), (1,))].item(),
            )
            pred_mask = predictor.predict(
                point_coords=np.array([[seed_x, seed_y]]),
                point_labels=np.array([1]),
                multimask_output=False,
            )[0][0].astype(np.uint8)

            iou_val = compute_instance_iou(pred_mask, gt_mask_np, label_id.item())
            label_id_int = int(label_id.item())
            per_class_ious.setdefault(label_id_int, []).append(iou_val)
            num_total_iou_calcs += 1

        num_processed += 1

    print(
        f"\n[INFO] Processed {num_processed} patches, skipped {num_skipped}. IoU calculations: {num_total_iou_calcs}"
    )

    csv_path = out_dir / (
        f"per_class_ious_subset_{args.subset_index}.csv"
        if args.total_subsets and args.subset_index is not None
        else "per_class_ious_all.csv"
    )
    print(f"\nSaving CSV to: {csv_path}")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "mean_iou", "std_iou", "count"])
        for cls_id, iou_list in per_class_ious.items():
            writer.writerow(
                [
                    cls_id,
                    float(np.mean(iou_list)),
                    float(np.std(iou_list)),
                    len(iou_list),
                ]
            )

    print(f"\nDone! All outputs saved in {out_dir}")


if __name__ == "__main__":
    main()
