#!/usr/bin/env python
"""
train_sam.py — DDP fine-tuning of SAM mask decoder on Kane County via TorchGeo
"""
import sys, os, argparse, random, csv
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
# import cv2 # Not strictly needed if PIL and matplotlib handle image ops
# from PIL import Image # Not strictly needed if PIL and matplotlib handle image ops
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import time # For timing

# === ADD MATPLOTLIB IMPORTS START ===
import matplotlib
matplotlib.use("Agg") # Use a non-interactive backend for headless environments
import matplotlib.pyplot as plt
# === ADD MATPLOTLIB IMPORTS END ===

# Expose repo root so imports from `data` and `prompted_kc` work
print("DEBUG: Setting up repo_root and sys.path...", flush=True)
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
print(f"DEBUG: sys.path updated with repo_root: {repo_root}", flush=True)

# TorchGeo
print("DEBUG: Importing TorchGeo components...", flush=True)
from torchgeo.datasets import NAIP, stack_samples
from torchgeo.samplers import RandomGeoSampler, Units
print("DEBUG: TorchGeo components imported.", flush=True)

# Local dataset subclass
print("DEBUG: Importing local dataset subclasses (PromptedKaneCounty, KaneCounty)...", flush=True)
from prompted_kc import PromptedKaneCounty
from data.kc import KaneCounty
print("DEBUG: Local dataset subclasses imported.", flush=True)

# SAM
print("DEBUG: Importing SAM components...", flush=True)
sys.path.append(str(repo_root / "segment_anything_source_code"))
from segment_anything.build_sam import sam_model_registry
from segment_anything.predictor import SamPredictor
print("DEBUG: SAM components imported.", flush=True)

# === DEFINE PLOT_EVERY_N_BATCHES CONSTANT START ===
PLOT_EVERY_N_BATCHES = 100  # Save a debug image every 100 batches
# === DEFINE PLOT_EVERY_N_BATCHES CONSTANT END ===

def parse_args():
    print("DEBUG: parse_args() called.", flush=True)
    p = argparse.ArgumentParser(description="DDP fine-tune SAM on KC with TorchGeo")
    # SAM checkpoint + outputs
    p.add_argument("--checkpoint", required=True, help="Path to SAM checkpoint .pth")
    p.add_argument("--output",     default="fine_tuned_sam.pth",    help="Where to save weights")
    p.add_argument("--csv-output", default="per_class_ious.csv",    help="CSV of per-class IoUs")
    # training hyperparams
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--batch-size",  type=int,   default=1)
    p.add_argument("--accum-steps", type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-5)
    p.add_argument("--num-workers", type=int,   default=4)
    # TorchGeo data
    p.add_argument("--naip-root",  required=True, help="Root directory of NAIP imagery")
    p.add_argument("--shape-path", required=True, help="Path to KC .gpkg file or .gdb directory/zip")
    p.add_argument("--layer-name", default="Basins", help="Layer name or index in GeoPackage/GDB")
    p.add_argument("--chip-size",  type=int, default=512, help="Patch size in pixels")
    # DDP
    p.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK",0)))
    print("DEBUG: parse_args() finished.", flush=True)
    return p.parse_args()

def main():
    print("DEBUG: main() started.", flush=True)
    args = parse_args()
    print(f"DEBUG: Arguments parsed: {args}", flush=True)

    # ── DDP init ────────────────────────────────────────────────────────
    print("DEBUG: Initializing Distributed Data Parallel (DDP)...", flush=True)
    os.environ.setdefault("MASTER_ADDR","localhost")
    os.environ.setdefault("MASTER_PORT", str(10000 + random.randrange(10000)))
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda:{args.local_rank}")
    rank  = dist.get_rank()
    world = dist.get_world_size()
    print(f"DEBUG: DDP Initialized. Rank: {rank}, World Size: {world}, Device: {device}", flush=True)

    # ── Logging / TB ────────────────────────────────────────────────────
    print("DEBUG: Setting up SummaryWriter and debug directory...", flush=True)
    writer = SummaryWriter("runs/sam_kc") if rank == 0 else None
    debug_dir = Path("debug_images") # This is where images will be saved
    if rank == 0:
        debug_dir.mkdir(exist_ok=True)
    print("DEBUG: SummaryWriter and debug directory setup complete.", flush=True)

    # ── Load & freeze SAM ───────────────────────────────────────────────
    print(f"DEBUG: Loading SAM model from checkpoint: {args.checkpoint}...", flush=True)
    sam = sam_model_registry["vit_h"](checkpoint=str(args.checkpoint))
    sam.to(device)
    print("DEBUG: SAM model loaded to device.", flush=True)
    print("DEBUG: Freezing SAM image encoder and prompt encoder parameters...", flush=True)
    for p in sam.image_encoder.parameters(): p.requires_grad = False
    for p in sam.prompt_encoder.parameters(): p.requires_grad = False
    sam.image_encoder.eval(); sam.prompt_encoder.eval(); sam.mask_decoder.train()
    print("DEBUG: SAM parameters frozen and modes set. Wrapping with DDP...", flush=True)
    sam = DDP(sam, device_ids=[args.local_rank], output_device=args.local_rank)
    print("DEBUG: SAM model wrapped with DDP.", flush=True)

    # ── Build TorchGeo dataset & sampler ────────────────────────────────
    print(f"DEBUG: Initializing NAIP dataset from naip-root: {args.naip_root}...", flush=True)
    naip_ds = NAIP(args.naip_root)
    print(f"DEBUG: NAIP dataset initialized. CRS: {naip_ds.crs}, Resolution: {naip_ds.res}", flush=True)

    print("DEBUG: Preparing idmap for KaneCounty labels...", flush=True)
    idmap = {name: id_val for id_val, name in KaneCounty.all_labels.items()}
    print(f"DEBUG: idmap created (string keys to int values): {idmap}", flush=True)

    print(f"DEBUG: CRITICAL STEP - Initializing PromptedKaneCounty dataset...", flush=True)
    print(f"DEBUG:   shape-path: {args.shape_path}", flush=True)
    print(f"DEBUG:   raw layer-name from args: '{args.layer_name}'", flush=True)
    print(f"DEBUG:   chip-size: {args.chip_size}", flush=True)

    layer_identifier_for_kc = args.layer_name
    try:
        layer_identifier_for_kc = int(args.layer_name)
        print(f"DEBUG: Interpreted --layer-name '{args.layer_name}' as INTEGER index: {layer_identifier_for_kc}", flush=True)
    except ValueError:
        print(f"DEBUG: Interpreting --layer-name '{args.layer_name}' as STRING name: '{layer_identifier_for_kc}'", flush=True)
        
    label_ds_configs = (layer_identifier_for_kc, idmap, args.chip_size, naip_ds.crs, naip_ds.res)
    
    print(f"DEBUG: Starting timer for PromptedKaneCounty initialization...", flush=True)
    start_time_label_ds = time.time()
    label_ds = PromptedKaneCounty(args.shape_path, label_ds_configs)
    end_time_label_ds = time.time()
    elapsed_time_label_ds = end_time_label_ds - start_time_label_ds
    print(f"DEBUG: CRITICAL STEP - PromptedKaneCounty dataset initialization FINISHED. Took {elapsed_time_label_ds:.2f} seconds.", flush=True)

    print("DEBUG: Combining NAIP and label datasets...", flush=True)
    combined = naip_ds & label_ds
    print(f"DEBUG: Combined dataset created. Attempting to get length...", flush=True)
    combined_len = len(combined)
    print(f"DEBUG: Combined dataset length: {combined_len}", flush=True)

    print("DEBUG: Initializing RandomGeoSampler...", flush=True)
    sampler_length = max(1, combined_len // world)
    print(f"DEBUG:   Sampler target length: {sampler_length}", flush=True)
    sampler = RandomGeoSampler(combined, size=args.chip_size, length=sampler_length, units=Units.PIXELS)
    print("DEBUG: RandomGeoSampler initialized.", flush=True)

    effective_num_workers = args.num_workers
    print(f"DEBUG: Initializing DataLoader with batch_size={args.batch_size}, num_workers={effective_num_workers}...", flush=True)
    loader = DataLoader(combined, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=stack_samples, pin_memory=True)
    print("DEBUG: DataLoader initialized.", flush=True)

    print("DEBUG: Setting up optimizer and GradScaler...", flush=True)
    optimizer = torch.optim.AdamW(sam.module.mask_decoder.parameters(), lr=args.lr)
    # Using recommended torch.amp.GradScaler for future compatibility
    scaler    = torch.cuda.amp.GradScaler()
    print("DEBUG: Optimizer and GradScaler setup complete.", flush=True)

    final_ious = defaultdict(list)
    print("DEBUG: Initialized final_ious defaultdict.", flush=True)

    print("DEBUG: === SETUP COMPLETE === Starting training loop... ===", flush=True)
    for epoch in range(args.epochs):
        print(f"DEBUG: Epoch {epoch+1}/{args.epochs} - Starting...", flush=True)
        sam.module.mask_decoder.train()
        running_loss = 0.0
        running_iou  = 0.0
        steps = 0 # Optimizer steps

        print(f"DEBUG: Epoch {epoch+1} - About to iterate over DataLoader (length: {len(loader)})...", flush=True)
        for batch_idx, batch in enumerate(loader):
            # `batch_idx` is the index of the current batch from the DataLoader
            print(f"DEBUG: Epoch {epoch+1} - Batch {batch_idx+1}/{len(loader)} received.", flush=True)
            imgs   = batch["image"].to(device)
            masks  = batch["mask"].to(device) # Ground truth masks
            points = batch["point"]
            B = imgs.size(0) # Actual batch size for this iteration
            print(f"DEBUG: Epoch {epoch+1} - Batch {batch_idx+1} - Data moved to device. Batch size: {B}", flush=True)

            loss_batch_accumulator = 0.0 # Accumulate loss for micro-batches

            # Using recommended torch.amp.autocast for future compatibility
            with torch.cuda.amp.autocast():
                for b in range(B): # Loop over items in the micro-batch
                    img_tensor_cpu = imgs[b].cpu()
                    if img_tensor_cpu.shape[0] == 4:
                        rgb_tensor_cpu = img_tensor_cpu[:3, :, :]
                    else:
                        rgb_tensor_cpu = img_tensor_cpu
                    
                    img_np_rgb = rgb_tensor_cpu.permute(1, 2, 0).numpy().astype(np.uint8)
                    
                    predictor = SamPredictor(sam.module) 
                    predictor.set_image(img_np_rgb)
                    emb = predictor.get_image_embedding().to(device)
                    
                    pt = torch.from_numpy(points[b]).to(device).unsqueeze(0).unsqueeze(0).float()
                    lbl = torch.ones((1,1), device=device) 
                    sp, dn = sam.module.prompt_encoder((pt,lbl), None, None) 
                    
                    low_res_masks, _ = sam.module.mask_decoder( 
                        image_embeddings=emb,
                        image_pe=sam.module.prompt_encoder.get_dense_pe().to(device), 
                        sparse_prompt_embeddings=sp,
                        dense_prompt_embeddings=dn,
                        multimask_output=False
                    )
                    
                    upscaled_masks = torch.nn.functional.interpolate(
                        low_res_masks, size=masks[b].shape[-2:], mode="bilinear", align_corners=False # Use H,W from mask
                    )
                    
                    # Ground truth for loss: binary mask (object presence)
                    # masks[b] might have shape [H, W] or [1, H, W]. Ensure it's [H,W] before >0
                    gt_binary_for_loss = (masks[b].squeeze() > 0).float().unsqueeze(0).unsqueeze(0) # Ensure [1,1,H,W] for BCE
                    
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(upscaled_masks, gt_binary_for_loss)
                    loss_batch_accumulator += loss

                    # For IoU and plotting:
                    pred_binary_mask = (torch.sigmoid(upscaled_masks) > 0.5).int().squeeze() # Get HxW binary mask
                    gt_mask_for_iou = masks[b].squeeze().int() # Get HxW integer ground truth mask

                    inter = (pred_binary_mask & gt_mask_for_iou).sum().item()
                    union = (pred_binary_mask | gt_mask_for_iou).sum().item()
                    iou = inter/union if union>0 else 1.0 # Or 0.0 if union is 0 and inter is 0
                    
                    # Get class from the ground truth mask at the prompt point for per-class IoU
                    # Ensure points[b] are valid indices for masks[b]
                    prompt_y, prompt_x = points[b][1], points[b][0] # points are [x,y]
                    cls = int(gt_mask_for_iou[prompt_y, prompt_x].item()) 
                    final_ious[cls].append(iou)
                    running_iou += iou

                    # === ADD PLOTTING CODE START ===
                    if rank == 0 and b == 0 and batch_idx % PLOT_EVERY_N_BATCHES == 0:
                        print(f"DEBUG: Generating debug plot for Epoch {epoch+1}, Batch {batch_idx+1}", flush=True)
                        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                        
                        # Pane 1: RGB Image
                        axes[0].imshow(img_np_rgb)
                        axes[0].set_title(f"Input RGB (Epoch {epoch+1}, Batch {batch_idx+1})")
                        axes[0].axis('off')

                        # Pane 2: True Label Mask
                        # KaneCounty.all_labels maps int IDs to string names. Max ID is 15.
                        true_mask_display = gt_mask_for_iou.cpu().numpy()
                        axes[1].imshow(true_mask_display, cmap='viridis', vmin=0, vmax=max(KaneCounty.all_labels.keys()))
                        axes[1].set_title("True Label Mask")
                        axes[1].axis('off')

                        # Pane 3: Predicted Mask
                        pred_mask_display = pred_binary_mask.cpu().numpy()
                        axes[2].imshow(pred_mask_display, cmap='gray') # Binary, so gray is fine
                        axes[2].set_title("Predicted Mask (Binary)")
                        axes[2].axis('off')

                        plt.tight_layout()
                        plot_filename = debug_dir / f"epoch{epoch+1}_batch{batch_idx+1}_item{b}.png"
                        try:
                            plt.savefig(plot_filename)
                            print(f"DEBUG: Saved debug plot to {plot_filename}", flush=True)
                        except Exception as e_plot:
                            print(f"DEBUG: ERROR saving plot {plot_filename}: {e_plot}", flush=True)
                        plt.close(fig) # Close the figure to free memory
                    # === ADD PLOTTING CODE END ===

            # Scale loss for gradient accumulation
            loss_to_scale = loss_batch_accumulator / args.accum_steps 
            scaler.scale(loss_to_scale).backward()
            
            # Accumulate running loss (use the scaled loss for apples-to-apples comparison with printed loss)
            running_loss += loss_to_scale.item() # Log the scaled loss

            if (batch_idx + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                steps += 1 # Increment optimizer steps count

                if rank==0 and steps % 10 == 0: # Log every 10 optimizer steps
                    # Average loss is running_loss / (number of items contributing to running_loss)
                    # Number of items = steps * B (if batch_size B is always constant)
                    # Or, more simply, average loss per optimizer step:
                    avg_l = running_loss / steps # running_loss already accounts for B and accum_steps
                    avg_i = running_iou / (steps * args.batch_size * args.accum_steps) # IoU is per item
                    print(f"[Epoch {epoch+1}] Opt_Step {steps} (Batch {batch_idx+1}) → loss {avg_l:.4f}, IoU {avg_i:.4f}", flush=True)


        print(f"DEBUG: Epoch {epoch+1} - Finished iterating over DataLoader.", flush=True)
        dist.barrier()
        print(f"DEBUG: Epoch {epoch+1} - Passed dist.barrier().", flush=True)
        if rank==0:
            # Calculate epoch averages
            # steps is total optimizer steps in this epoch
            # total items processed = steps * args.batch_size * args.accum_steps
            if steps > 0: # Avoid division by zero if epoch had no steps
                avg_l_epoch = running_loss / steps 
                avg_i_epoch = running_iou / (steps * args.batch_size * args.accum_steps)
                if writer: writer.add_scalar("Loss/train", avg_l_epoch, epoch)
                if writer: writer.add_scalar("IoU/train",  avg_i_epoch, epoch)
                print(f"*** Epoch {epoch+1} done: loss {avg_l_epoch:.4f}, IoU {avg_i_epoch:.4f}", flush=True)
            else:
                print(f"*** Epoch {epoch+1} done: No optimizer steps taken.", flush=True)

        print(f"DEBUG: Epoch {epoch+1}/{args.epochs} - Finished.", flush=True)

    # ── Finalize ────────────────────────────────────────────────────────
    print("DEBUG: Training loop finished. Finalizing...", flush=True)
    if rank==0:
        print(f"DEBUG: Rank 0 saving model to {args.output} and CSV to {args.csv_output}...", flush=True)
        torch.save(sam.module.state_dict(), args.output)
        with open(args.csv_output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class_id","mean_iou","std_iou","count"])
            for cls_id_val, ious_list in final_ious.items(): # Renamed variables for clarity
                if ious_list: 
                    w.writerow([cls_id_val, np.mean(ious_list), np.std(ious_list), len(ious_list)])
                else: 
                    w.writerow([cls_id_val, 0.0, 0.0, 0])
        if writer: writer.close()
        print("DEBUG: Rank 0 model and CSV saved. Writer closed.", flush=True)

    print("DEBUG: Destroying DDP process group...", flush=True)
    dist.destroy_process_group()
    print("DEBUG: DDP process group destroyed. main() finished.", flush=True)

if __name__ == "__main__":
    print("DEBUG: Script execution started (__name__ == '__main__').", flush=True)
    main()