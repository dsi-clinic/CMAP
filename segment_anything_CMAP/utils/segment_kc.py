#!/usr/bin/env python

import argparse
import csv
import os

import matplotlib
import numpy as np
import torch

# Use a non-interactive backend for matplotlib (headless mode)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from segment_anything_source_code import SamPredictor, sam_model_registry
from torch.utils.data import DataLoader
from torchgeo.datasets import NAIP, stack_samples

from configs import config
from data.kc import KaneCounty
from data.sampler import BalancedRandomBatchGeoSampler


def compute_instance_iou(pred_mask, gt_mask, instance_id):
    """Compute IoU for one predicted mask vs. one ground-truth label ID.
    pred_mask, gt_mask: same shape
    instance_id: the ground-truth label ID (class/instance) we care about
    """
    gt_instance = (gt_mask == instance_id).astype(np.uint8)
    pred_instance = (pred_mask > 0).astype(np.uint8)  # 1=object, 0=bg
    intersection = np.logical_and(gt_instance, pred_instance).sum()
    union = np.logical_or(gt_instance, pred_instance).sum()
    return float(intersection) / union if union != 0 else 0.0


def main():
    """ProcessNAIP imagery by running Segment Anything (SAM) on each patch,
    optionally in parallel when using an HPC array job.

    This function:
      1. Loads imagery and a corresponding shapefile to build a dataset.
      2. Applies a BalancedRandomBatchGeoSampler to enumerate patches (i.e., subsets of the imagery).
      3. Loops over each patch, optionally subdividing the patches among multiple array tasks:
         - If using Slurm, you can pass --subset_index=X and --total_subsets=Y to have each
           sub-task handle a different portion of the patches (parallel processing).
         - Patches not in the sub-task's assigned range are skipped.
      4. For each valid label in a patch, picks a random pixel and runs the SAM model to obtain
         a segmentation mask. Then it computes the IoU against the ground-truth.
      5. Optionally saves a limited number of debug plots (up to MAX_SAVED_PLOTS).
      6. Aggregates all IoU results by class, saving a CSV (per array sub-task or overall) in
         one output folder.

    Args:
        --max_samples (int): If >0, limit total patch processing to that many; if -1, process all.
        --subset_index (int): This sub-task's index if running multiple parallel jobs.
        --total_subsets (int): Total number of parallel sub-tasks (used to partition patch indices).

    Note:
        - No previous runs are deleted, so repeated calls may accumulate outputs unless
          manually cleared.
        - If using an HPC array, each sub-task writes its partial CSV in the same folder but
          with a unique name (e.g., per_class_ious_subset_{index}.csv).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Process up to this many patches; -1 for all.",
    )
    parser.add_argument(
        "--subset_index",
        type=int,
        default=None,
        help="Index of subset if splitting data among multiple jobs.",
    )
    parser.add_argument(
        "--total_subsets",
        type=int,
        default=None,
        help="Total subsets to split the data (used with subset_index).",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Output directory setup (NO deletion of old runs => Aggregator will including old data if not deleted first!!)
    # -------------------------------------------------------------------------
    home_dir = os.path.expanduser("~")
    base_out_dir = os.path.join(home_dir, "CMAP", "segment_anything_CMAP", "kc_sam_outputs")
    os.makedirs(base_out_dir, exist_ok=True)

    job_id = os.getenv("SLURM_JOB_ID", "local")
    run_folder_name = f"kc_sam_run_{job_id}"
    out_dir = os.path.join(base_out_dir, run_folder_name)
    os.makedirs(out_dir, exist_ok=True)

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"[INFO] Using single output directory for all tasks:\n  {out_dir}")

    # -------------------------------------------------------------------------
    # 1) Load dataset
    # -------------------------------------------------------------------------
    naip_dataset = NAIP("/net/projects/cmap/data/KC-images")
    shape_path = os.path.join(config.KC_SHAPE_ROOT, config.KC_SHAPE_FILENAME)

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

    try:
        total_samples = len(plot_dataloader)
    except TypeError:
        total_samples = None

    print("[INFO] Loading SAM model...")
    sam_checkpoint = os.path.join(home_dir, "CMAP", "segment_anything_source_code", "sam_vit_h.pth")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)

    # -------------------------------------------------------------------------
    # 2) Loop over patches, apply SAM
    # -------------------------------------------------------------------------
    per_class_ious = {}
    num_processed = 0
    num_skipped = 0
    num_total_iou_calcs = 0

    MAX_SAVED_PLOTS = 10  # number of plots for each parallel job to save
    saved_plots_count = 0

    print("[INFO] Starting segmentation loop...")

    for sample_idx, batch in enumerate(plot_dataloader):
        # (a) If max_samples != -1, limit total
        if args.max_samples != -1 and sample_idx >= args.max_samples:
            break

        # (b) Partition logic for array
        if (
            args.total_subsets
            and args.subset_index is not None
            and total_samples is not None
        ):
            subset_size = total_samples // args.total_subsets
            start_idx = args.subset_index * subset_size
            end_idx = (args.subset_index + 1) * subset_size
            # last subset picks up remainder
            if args.subset_index == args.total_subsets - 1:
                end_idx = total_samples
            if not (start_idx <= sample_idx < end_idx):
                continue

        # Progress print
        if total_samples is not None:
            pct = 100.0 * sample_idx / total_samples
            print(
                f"Processing patch {sample_idx+1}/{total_samples} ({pct:.1f}% complete)."
            )
        else:
            print(f"Processing patch {sample_idx}...")

        # Extract image & mask from batch
        img_tensor = batch["image"][0]
        mask_tensor = batch["mask"][0]
        if mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
            mask_tensor = mask_tensor.squeeze(0)

        # Identify valid labels in ground truth
        valid_labels = torch.unique(mask_tensor)
        valid_labels = valid_labels[valid_labels > 0]
        if len(valid_labels) == 0:
            num_skipped += 1
            continue

        # Convert to NumPy for SAM
        img_np = img_tensor[:3].cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0)).astype(np.uint8)
        # print the tuple; make sure we're getting the best mask
        predictor.set_image(img_np)
        gt_mask_np = mask_tensor.cpu().numpy()

        # Segment each valid label
        for label_id in valid_labels:
            ys, xs = torch.where(mask_tensor == label_id)
            if len(xs) == 0:
                continue

            # Pick a random seed from that label
            rand_idx = torch.randint(0, len(xs), (1,)).item()
            seed_x, seed_y = xs[rand_idx].item(), ys[rand_idx].item()

            # Run SAM
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[seed_x, seed_y]]),
                point_labels=np.array([1]),
                multimask_output=False,
            )
            pred_mask = masks[0].astype(np.uint8)

            # Compute IoU for that label
            iou_val = compute_instance_iou(pred_mask, gt_mask_np, label_id.item())
            label_id_int = int(label_id.item())
            if label_id_int not in per_class_ious:
                per_class_ious[label_id_int] = []
            per_class_ious[label_id_int].append(iou_val)
            num_total_iou_calcs += 1

            # Optionally save some debug plots
            if saved_plots_count < MAX_SAVED_PLOTS:
                saved_plots_count += 1
                fig, axs = plt.subplots(1, 3, figsize=(16, 5))

                axs[0].imshow(img_np)
                axs[0].scatter(seed_x, seed_y, color="red", s=40, edgecolor="black")
                axs[0].set_title("RGB + Seed")
                axs[0].axis("off")

                axs[1].imshow(gt_mask_np, cmap="gray")
                axs[1].scatter(seed_x, seed_y, color="red", s=40, edgecolor="black")
                axs[1].set_title("GT Mask")
                axs[1].axis("off")

                axs[2].imshow(pred_mask, cmap="gray")
                axs[2].scatter(seed_x, seed_y, color="red", s=40, edgecolor="black")
                axs[2].set_title(f"Pred Mask (IoU={iou_val:.2f})")
                axs[2].axis("off")

                plt.tight_layout()

                si = args.subset_index if args.subset_index is not None else 0
                plot_fname = f"task{si}_patch_{sample_idx}_label_{label_id_int}_{saved_plots_count}.png"
                out_plot_path = os.path.join(plots_dir, plot_fname)
                plt.savefig(out_plot_path, dpi=150)
                plt.close(fig)

        num_processed += 1

    print("\n--- Inference Complete ✅---")
    print(
        f"Processed {num_processed} patches (skipped {num_skipped})."
    )  # "skipped" patches == those outside of sub-task's assigned range
    print(f"Total per-label IoU calculations: {num_total_iou_calcs}")

    # Summarize IoU
    print("\nPer-class IoU Summary:")
    for cls_id, iou_list in per_class_ious.items():
        mean_iou = np.mean(iou_list)
        std_iou = np.std(iou_list)
        print(f"  Class {cls_id}: Mean IoU={mean_iou:.4f}, Std={std_iou:.4f}")

    # Write CSV
    if args.total_subsets and args.subset_index is not None:
        csv_name = f"per_class_ious_subset_{args.subset_index}.csv"
    else:
        csv_name = "per_class_ious_all.csv"

    csv_path = os.path.join(out_dir, csv_name)
    print(f"\nSaving CSV to: {csv_path}")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "mean_iou", "std_iou", "count"])
        for cls_id, iou_list in per_class_ious.items():
            mean_iou = float(np.mean(iou_list))
            std_iou = float(np.std(iou_list))
            count = len(iou_list)
            writer.writerow([cls_id, mean_iou, std_iou, count])

    print(f"\nDone! All tasks wrote to the single folder:\n  {out_dir}")


if __name__ == "__main__":
    main()
