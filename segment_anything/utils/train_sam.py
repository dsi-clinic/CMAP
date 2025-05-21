#!/usr/bin/env python
"""train_sam.py — DDP fine-tuning of SAM mask decoder on Kane County via TorchGeo"""

import argparse
import csv
import logging  # Import the logging module
import os
import secrets  # for cryptographically secure random numbers
import time  # For timing
from collections import defaultdict
from pathlib import Path

import einops  # for tensor manipulation
import matplotlib
import numpy as np
import torch
import torch.distributed as dist
from prompted_kc import PromptedKaneCounty

# import cv2 # Not strictly needed if PIL and matplotlib handle image ops
# from PIL import Image # Not strictly needed if PIL and matplotlib handle image ops
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchgeo.datasets import NAIP, stack_samples
from torchgeo.samplers import RandomGeoSampler, Units

from data.kc import KaneCounty
from segment_anything.build_sam import sam_model_registry
from segment_anything.predictor import SamPredictor

matplotlib.use("Agg")  # Use a non-interactive backend for headless environments
import matplotlib.pyplot as plt

# Expose repo root so imports from `data` and `prompted_kc` work
# print("DEBUG: Setting up repo_root and sys.path...", flush=True)
# repo_root = Path(__file__).resolve().parents[2]
# sys.path.insert(0, str(repo_root))
# print(f"DEBUG: sys.path updated with repo_root: {repo_root}", flush=True)


# === DEFINE PLOT_EVERY_N_BATCHES CONSTANT START ===
PLOT_EVERY_N_BATCHES = 100  # save a debug image every 100 batches
# === DEFINE PLOT_EVERY_N_BATCHES CONSTANT END ===
RGBA_CHANNELS = 4  # number of channels in an RGBA image
SIGMOID_THRESHOLD = 0.5  # threshold for converting sigmoid output to binary mask


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="DDP fine-tune SAM on KC with TorchGeo")
    # SAM checkpoint + outputs
    p.add_argument("--checkpoint", required=True, help="Path to SAM checkpoint .pth")
    p.add_argument(
        "--output-dir",
        default="sam_finetune_output",
        help="Directory to save outputs (model, logs, csv, debug images)",
    )
    # training hyperparams
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=4)
    # TorchGeo data
    p.add_argument("--naip-root", required=True, help="Root directory of NAIP imagery")
    p.add_argument(
        "--shape-path",
        required=True,
        help="Path to KC .gpkg file or .gdb directory/zip",
    )
    p.add_argument(
        "--layer-name", default="Basins", help="Layer name or index in GeoPackage/GDB"
    )
    p.add_argument("--chip-size", type=int, default=512, help="Patch size in pixels")
    # DDP
    p.add_argument(
        "--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0))
    )
    print("DEBUG: parse_args() finished.", flush=True)
    return p.parse_args()


def main():
    """Main training script."""
    print("DEBUG: main() started.", flush=True)
    args = parse_args()
    print(f"DEBUG: Arguments parsed: {args}", flush=True)

    # ── DDP init ────────────────────────────────────────────────────────
    print("DEBUG: Initializing Distributed Data Parallel (DDP)...", flush=True)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(10000 + secrets.randbelow(10000)))
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda:{args.local_rank}")
    rank = dist.get_rank()
    world = dist.get_world_size()
    print(
        f"DEBUG: DDP Initialized. Rank: {rank}, World Size: {world}, Device: {device}",
        flush=True,
    )

    # Create output directory on rank 0
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    model_output_path = output_dir / "fine_tuned_sam.pth"
    csv_output_path = output_dir / "per_class_ious.csv"
    log_file_path = output_dir / "training.log"
    debug_images_dir = output_dir / "debug_images"
    tensorboard_log_dir = output_dir / "runs" / "sam_kc"

    log_level = logging.DEBUG  # Or use logging.INFO for less verbosity
    handlers_list = []
    if rank == 0:
        handlers_list.append(logging.StreamHandler())  # stdout
        handlers_list.append(logging.FileHandler(log_file_path))  # log file
    else:
        handlers_list.append(logging.NullHandler())

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers_list,
        force=True,  # Requires Python 3.8+
    )

    # Suppress verbose logs from specific libraries
    libraries_to_quiet = [
        "rasterio",
        "fiona",
        "matplotlib",
    ]  # Add other libraries as needed
    for lib_name in libraries_to_quiet:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(logging.WARNING)

    # ── Logging / TB ────────────────────────────────────────────────────
    logging.debug("Setting up SummaryWriter and debug directory...")
    writer = SummaryWriter(log_dir=str(tensorboard_log_dir)) if rank == 0 else None
    if rank == 0:
        debug_images_dir.mkdir(exist_ok=True)
    logging.debug("SummaryWriter and debug directory setup complete.")

    # ── Load & freeze SAM ───────────────────────────────────────────────
    logging.debug(f"Loading SAM model from checkpoint: {args.checkpoint}...")
    sam = sam_model_registry["vit_h"](checkpoint=str(args.checkpoint))
    sam.to(device)
    logging.debug("SAM model loaded to device.")
    logging.debug("Freezing SAM image encoder and prompt encoder parameters...")
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False
    sam.image_encoder.eval()
    sam.prompt_encoder.eval()
    sam.mask_decoder.train()
    logging.debug("SAM parameters frozen and modes set. Wrapping with DDP...")
    sam = DDP(sam, device_ids=[args.local_rank], output_device=args.local_rank)
    logging.debug("SAM model wrapped with DDP.")

    # ── Build TorchGeo dataset & sampler ────────────────────────────────
    logging.debug(f"Initializing NAIP dataset from naip-root: {args.naip_root}...")
    naip_ds = NAIP(args.naip_root)
    logging.debug(
        f"NAIP dataset initialized. CRS: {naip_ds.crs}, Resolution: {naip_ds.res}"
    )

    logging.debug("Preparing idmap for KaneCounty labels...")
    idmap = {name: id_val for id_val, name in KaneCounty.all_labels.items()}
    logging.debug(f"idmap created (string keys to int values): {idmap}")

    logging.info("CRITICAL STEP - Initializing PromptedKaneCounty dataset...")
    logging.debug(f"  shape-path: {args.shape_path}")
    logging.debug(f"  raw layer-name from args: '{args.layer_name}'")
    logging.debug(f"  chip-size: {args.chip_size}")

    layer_identifier_for_kc = args.layer_name
    try:
        layer_identifier_for_kc = int(args.layer_name)
        logging.debug(
            f"Interpreted --layer-name '{args.layer_name}' as INTEGER index: {layer_identifier_for_kc}"
        )
    except ValueError:
        logging.debug(
            f"Interpreting --layer-name '{args.layer_name}' as STRING name: '{layer_identifier_for_kc}'"
        )

    label_ds_configs = (
        layer_identifier_for_kc,
        idmap,
        args.chip_size,
        naip_ds.crs,
        naip_ds.res,
    )

    logging.debug("Starting timer for PromptedKaneCounty initialization...")
    start_time_label_ds = time.time()
    label_ds = PromptedKaneCounty(args.shape_path, label_ds_configs)
    end_time_label_ds = time.time()
    elapsed_time_label_ds = end_time_label_ds - start_time_label_ds
    logging.info(
        f"CRITICAL STEP - PromptedKaneCounty dataset initialization FINISHED. Took {elapsed_time_label_ds:.2f} seconds."
    )

    logging.debug("Combining NAIP and label datasets...")
    combined = naip_ds & label_ds
    logging.debug("Combined dataset created. Attempting to get length...")
    combined_len = len(combined)
    logging.debug(f"Combined dataset length: {combined_len}")

    logging.debug("Initializing RandomGeoSampler...")
    sampler_length = max(1, combined_len // world)
    logging.debug(f"  Sampler target length: {sampler_length}")
    sampler = RandomGeoSampler(
        combined, size=args.chip_size, length=sampler_length, units=Units.PIXELS
    )
    logging.debug("RandomGeoSampler initialized.")

    effective_num_workers = args.num_workers
    logging.debug(
        f"Initializing DataLoader with batch_size={args.batch_size}, num_workers={effective_num_workers}..."
    )
    loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=stack_samples,
        pin_memory=True,
    )
    logging.debug("DataLoader initialized.")

    logging.debug("Setting up optimizer and GradScaler...")
    optimizer = torch.optim.AdamW(sam.module.mask_decoder.parameters(), lr=args.lr)
    # Using recommended torch.amp.GradScaler for future compatibility
    scaler = torch.cuda.amp.GradScaler()
    logging.debug("Optimizer and GradScaler setup complete.")

    final_ious = defaultdict(list)
    logging.debug("Initialized final_ious defaultdict.")

    logging.info("=== SETUP COMPLETE === Starting training loop... ===")
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs} - Starting...")
        sam.module.mask_decoder.train()
        running_loss = 0.0
        running_iou = 0.0
        steps = 0  # optimizer steps

        logging.debug(
            f"Epoch {epoch+1} - About to iterate over DataLoader (length: {len(loader)})..."
        )
        for batch_idx, batch in enumerate(loader):
            # `batch_idx` is the index of the current batch from the DataLoader
            logging.debug(
                f"Epoch {epoch+1} - Batch {batch_idx+1}/{len(loader)} received."
            )
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)  # ground truth masks
            points = batch["point"]
            B = imgs.size(0)  # actual batch size for this iteration
            logging.debug(
                f"Epoch {epoch+1} - Batch {batch_idx+1} - Data moved to device. Batch size: {B}"
            )

            loss_batch_accumulator = 0.0  # accumulate loss for micro-batches

            # Using recommended torch.amp.autocast for future compatibility
            with torch.cuda.amp.autocast():
                for b in range(B):  # loop over items in the micro-batch
                    img_tensor_cpu = imgs[b].cpu()
                    if img_tensor_cpu.shape[0] == RGBA_CHANNELS:
                        rgb_tensor_cpu = img_tensor_cpu[:3, :, :]
                    else:
                        rgb_tensor_cpu = img_tensor_cpu

                    img_np_rgb = (
                        einops.rearrange(rgb_tensor_cpu, "c h w -> h w c")
                        .numpy()
                        .astype(np.uint8)
                    )

                    predictor = SamPredictor(sam.module)
                    predictor.set_image(img_np_rgb)
                    emb = predictor.get_image_embedding().to(device)

                    pt = (
                        torch.from_numpy(points[b])
                        .to(device)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .float()
                    )
                    lbl = torch.ones((1, 1), device=device)
                    sp, dn = sam.module.prompt_encoder((pt, lbl), None, None)

                    low_res_masks, _ = sam.module.mask_decoder(
                        image_embeddings=emb,
                        image_pe=sam.module.prompt_encoder.get_dense_pe().to(device),
                        sparse_prompt_embeddings=sp,
                        dense_prompt_embeddings=dn,
                        multimask_output=False,
                    )

                    upscaled_masks = torch.nn.functional.interpolate(
                        low_res_masks,
                        size=masks[b].shape[-2:],
                        mode="bilinear",
                        align_corners=False,  # use H,W from mask
                    )

                    # Ground truth for loss: binary mask (object presence)
                    # masks[b] might have shape [H, W] or [1, H, W]. Ensure it's [H,W] before >0
                    gt_binary_for_loss = (
                        (masks[b].squeeze() > 0).float().unsqueeze(0).unsqueeze(0)
                    )  # ensure [1,1,H,W] for BCE

                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        upscaled_masks, gt_binary_for_loss
                    )
                    loss_batch_accumulator += loss

                    # For IoU and plotting:
                    pred_binary_mask = (
                        (torch.sigmoid(upscaled_masks) > SIGMOID_THRESHOLD)
                        .int()
                        .squeeze()
                    )  # get HxW binary mask
                    gt_mask_for_iou = (
                        masks[b].squeeze().int()
                    )  # get HxW integer ground truth mask

                    inter = (pred_binary_mask & gt_mask_for_iou).sum().item()
                    union = (pred_binary_mask | gt_mask_for_iou).sum().item()
                    iou = (
                        inter / union if union > 0 else 1.0
                    )  # or 0.0 if union is 0 and inter is 0

                    # Get class from the ground truth mask at the prompt point for per-class IoU
                    # Ensure points[b] are valid indices for masks[b]
                    prompt_y, prompt_x = points[b][1], points[b][0]  # points are [x,y]
                    cls = int(gt_mask_for_iou[prompt_y, prompt_x].item())
                    final_ious[cls].append(iou)
                    running_iou += iou

                    # === ADD PLOTTING CODE START ===
                    if rank == 0 and b == 0 and batch_idx % PLOT_EVERY_N_BATCHES == 0:
                        logging.debug(
                            f"Generating debug plot for Epoch {epoch+1}, Batch {batch_idx+1}"
                        )
                        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                        # Pane 1: RGB Image
                        axes[0].imshow(img_np_rgb)
                        axes[0].set_title(
                            f"input rgb (epoch {epoch+1}, batch {batch_idx+1})"
                        )
                        axes[0].axis("off")

                        # Pane 2: True Label Mask
                        # KaneCounty.all_labels maps int IDs to string names. Max ID is 15.
                        true_mask_display = gt_mask_for_iou.cpu().numpy()
                        axes[1].imshow(
                            true_mask_display,
                            cmap="viridis",
                            vmin=0,
                            vmax=max(KaneCounty.all_labels.keys()),
                        )
                        axes[1].set_title("true label mask")
                        axes[1].axis("off")

                        # Pane 3: Predicted Mask
                        pred_mask_display = pred_binary_mask.cpu().numpy()
                        axes[2].imshow(
                            pred_mask_display, cmap="gray"
                        )  # binary, so gray is fine
                        axes[2].set_title("predicted mask (binary)")
                        axes[2].axis("off")

                        plt.tight_layout()
                        plot_filename = (
                            debug_images_dir
                            / f"epoch{epoch+1}_batch{batch_idx+1}_item{b}.png"
                        )
                        try:
                            plt.savefig(plot_filename)
                            logging.debug(f"Saved debug plot to {plot_filename}")
                        except Exception as e_plot:
                            logging.error(
                                f"ERROR saving plot {plot_filename}: {e_plot}"
                            )
                        plt.close(fig)  # close the figure to free memory
                    # === ADD PLOTTING CODE END ===

            # Scale loss for gradient accumulation
            loss_to_scale = loss_batch_accumulator / args.accum_steps
            scaler.scale(loss_to_scale).backward()

            # Accumulate running loss (use the scaled loss for apples-to-apples comparison with printed loss)
            running_loss += loss_to_scale.item()  # log the scaled loss

            if (batch_idx + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                steps += 1  # increment optimizer steps count

                if rank == 0 and steps % 10 == 0:  # log every 10 optimizer steps
                    # Average loss is running_loss / (number of items contributing to running_loss)
                    # Number of items = steps * B (if batch_size B is always constant)
                    # Or, more simply, average loss per optimizer step:
                    avg_l = (
                        running_loss / steps
                    )  # running_loss already accounts for B and accum_steps
                    avg_i = running_iou / (
                        steps * args.batch_size * args.accum_steps
                    )  # iou is per item
                    logging.info(
                        f"[Epoch {epoch+1}] Opt_Step {steps} (Batch {batch_idx+1}) → loss {avg_l:.4f}, IoU {avg_i:.4f}"
                    )

        logging.debug(f"Epoch {epoch+1} - Finished iterating over DataLoader.")
        dist.barrier()
        logging.debug(f"Epoch {epoch+1} - Passed dist.barrier().")
        if rank == 0:
            # Calculate epoch averages
            # steps is total optimizer steps in this epoch
            # total items processed = steps * args.batch_size * args.accum_steps
            if steps > 0:  # avoid division by zero if epoch had no steps
                avg_l_epoch = running_loss / steps
                avg_i_epoch = running_iou / (steps * args.batch_size * args.accum_steps)
                if writer:
                    writer.add_scalar("Loss/train", avg_l_epoch, epoch)
                if writer:
                    writer.add_scalar("IoU/train", avg_i_epoch, epoch)
                logging.info(
                    f"*** Epoch {epoch+1} done: loss {avg_l_epoch:.4f}, IoU {avg_i_epoch:.4f}"
                )
            else:
                logging.info(f"*** Epoch {epoch+1} done: No optimizer steps taken.")

        logging.info(f"Epoch {epoch+1}/{args.epochs} - Finished.")

    # ── Finalize ────────────────────────────────────────────────────────
    logging.info("Training loop finished. Finalizing...")
    if rank == 0:
        logging.info(
            f"Rank 0 saving model to {model_output_path} and CSV to {csv_output_path}..."
        )
        torch.save(sam.module.state_dict(), model_output_path)
        with csv_output_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class_id", "mean_iou", "std_iou", "count"])
            for (
                cls_id_val,
                ious_list,
            ) in final_ious.items():  # renamed variables for clarity
                if ious_list:
                    w.writerow(
                        [
                            cls_id_val,
                            np.mean(ious_list),
                            np.std(ious_list),
                            len(ious_list),
                        ]
                    )
                else:
                    w.writerow([cls_id_val, 0.0, 0.0, 0])
        if writer:
            writer.close()
        logging.info("Rank 0 model and CSV saved. Writer closed.")

    logging.debug("Destroying DDP process group...")
    dist.destroy_process_group()
    logging.info("DDP process group destroyed. main() finished.")


if __name__ == "__main__":
    main()
