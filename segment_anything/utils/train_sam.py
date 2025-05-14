#!/usr/bin/env python
"""
train_sam.py — DDP fine-tuning of SAM mask decoder on Kane County via TorchGeo
"""
import sys, os, argparse, random, csv
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import cv2
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

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
    p.add_argument("--shape-path", required=True, help="Path to KC .gpkg file") # Original help text
    p.add_argument("--layer-name", default="Basins", help="Layer name in GeoPackage") # Original help text
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
    debug_dir = Path("debug_images")
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
    idmap = {k: k for k in KaneCounty.all_labels.keys()}
    print(f"DEBUG: idmap created: {idmap}", flush=True)

    print(f"DEBUG: CRITICAL STEP - Initializing PromptedKaneCounty dataset...", flush=True)
    print(f"DEBUG:   shape-path: {args.shape_path}", flush=True)
    print(f"DEBUG:   layer-name: {args.layer_name}", flush=True)
    print(f"DEBUG:   chip-size: {args.chip_size}", flush=True)
    label_ds_configs = (args.layer_name, idmap, args.chip_size, naip_ds.crs, naip_ds.res)
    label_ds = PromptedKaneCounty(
        args.shape_path,
        label_ds_configs
    )
    print("DEBUG: CRITICAL STEP - PromptedKaneCounty dataset initialization FINISHED.", flush=True)
    # If the script hangs BEFORE "FINISHED", the problem is inside PromptedKaneCounty or KaneCounty __init__

    print("DEBUG: Combining NAIP and label datasets...", flush=True)
    combined = naip_ds & label_ds
    # len(combined) can trigger dataset indexing/length calculation which might be slow
    print(f"DEBUG: Combined dataset created. Attempting to get length...", flush=True)
    combined_len = len(combined)
    print(f"DEBUG: Combined dataset length: {combined_len}", flush=True)

    print("DEBUG: Initializing RandomGeoSampler...", flush=True)
    sampler_length = max(1, combined_len // world)
    print(f"DEBUG:   Sampler target length: {sampler_length}", flush=True)
    sampler = RandomGeoSampler(
        combined,
        size=args.chip_size,
        length=sampler_length,
        units=Units.PIXELS
    )
    print("DEBUG: RandomGeoSampler initialized.", flush=True)

    print(f"DEBUG: Initializing DataLoader with batch_size={args.batch_size}, num_workers={args.num_workers}...", flush=True)
    loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=stack_samples,
        pin_memory=True,
    )
    print("DEBUG: DataLoader initialized.", flush=True)

    # ── Optimizer & scaler ───────────────────────────────────────────────
    print("DEBUG: Setting up optimizer and GradScaler...", flush=True)
    optimizer = torch.optim.AdamW(sam.module.mask_decoder.parameters(), lr=args.lr)
    scaler    = torch.cuda.amp.GradScaler()
    print("DEBUG: Optimizer and GradScaler setup complete.", flush=True)

    final_ious = defaultdict(list)
    print("DEBUG: Initialized final_ious defaultdict.", flush=True)

    print("DEBUG: === SETUP COMPLETE === Starting training loop... ===", flush=True)
    # ── Training loop ───────────────────────────────────────────────────
    for epoch in range(args.epochs):
        print(f"DEBUG: Epoch {epoch+1}/{args.epochs} - Starting...", flush=True)
        sam.module.mask_decoder.train()
        running_loss = 0.0
        running_iou  = 0.0
        steps = 0

        print(f"DEBUG: Epoch {epoch+1} - About to iterate over DataLoader (length: {len(loader)})...", flush=True)
        for batch_idx, batch in enumerate(loader):
            print(f"DEBUG: Epoch {epoch+1} - Batch {batch_idx+1}/{len(loader)} received.", flush=True)
            imgs   = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            points = batch["point"]
            B = imgs.size(0)
            print(f"DEBUG: Epoch {epoch+1} - Batch {batch_idx+1} - Data moved to device. Batch size: {B}", flush=True)

            loss_batch = 0.0
            # print(f"DEBUG: Epoch {epoch+1} - Batch {batch_idx+1} - Starting torch.cuda.amp.autocast()...", flush=True)
            with torch.cuda.amp.autocast():
                for b in range(B):
                    # print(f"DEBUG: Epoch {epoch+1} - Batch {batch_idx+1} - Processing item {b+1}/{B}...", flush=True)
                    # 1) embed
                    img_np = imgs[b].permute(1,2,0).cpu().numpy().astype(np.uint8)
                    predictor = SamPredictor(sam.module) # Use sam.module as sam is DDP-wrapped
                    predictor.set_image(img_np)
                    emb = predictor.get_image_embedding().to(device)
                    # 2) prompt
                    pt = torch.from_numpy(points[b]).to(device).unsqueeze(0).unsqueeze(0).float()
                    lbl = torch.ones((1,1), device=device)
                    sp, dn = sam.module.prompt_encoder((pt,lbl), None, None) # Use sam.module
                    # 3) decode
                    low, _ = sam.module.mask_decoder( # Use sam.module
                        image_embeddings=emb,
                        image_pe=sam.module.prompt_encoder.get_dense_pe().to(device), # Use sam.module
                        sparse_prompt_embeddings=sp,
                        dense_prompt_embeddings=dn,
                        multimask_output=False
                    )
                    up = torch.nn.functional.interpolate(
                        low, size=masks[b].shape, mode="bilinear", align_corners=False
                    )
                    gt = (masks[b]>0).float().unsqueeze(0).unsqueeze(0)
                    l  = torch.nn.functional.binary_cross_entropy_with_logits(up, gt)
                    loss_batch += l

                    # IoU
                    pred_bin = (up.sigmoid()>0.5).int().squeeze()
                    inter = (pred_bin & masks[b].int()).sum().item()
                    union = (pred_bin | masks[b].int()).sum().item()
                    iou = inter/union if union>0 else 1.0
                    cls = int(masks[b][points[b][1], points[b][0]].item()) # Original cls calculation
                    final_ious[cls].append(iou)
                    running_iou += iou

                # normalize by accumulation
                loss_batch = loss_batch / (args.accum_steps * B)
            # print(f"DEBUG: Epoch {epoch+1} - Batch {batch_idx+1} - amp.autocast() finished. Loss_batch: {loss_batch.item()}", flush=True)

            # print(f"DEBUG: Epoch {epoch+1} - Batch {batch_idx+1} - Scaling loss and calling backward()...", flush=True)
            scaler.scale(loss_batch).backward()
            steps += 1

            if steps % args.accum_steps == 0:
                # print(f"DEBUG: Epoch {epoch+1} - Batch {batch_idx+1} - Optimizer step and update (accum_steps reached).", flush=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss_batch.item()

            if rank==0 and steps%10==0:
                avg_l = running_loss/steps
                avg_i = running_iou/(steps*B)
                print(f"[Epoch {epoch+1}] Step {steps} (Batch {batch_idx+1}) → loss {avg_l:.4f}, IoU {avg_i:.4f}", flush=True)

        print(f"DEBUG: Epoch {epoch+1} - Finished iterating over DataLoader.", flush=True)
        dist.barrier()
        print(f"DEBUG: Epoch {epoch+1} - Passed dist.barrier().", flush=True)
        if rank==0:
            avg_l_epoch = running_loss/steps
            avg_i_epoch = running_iou/(steps*B)
            if writer: writer.add_scalar("Loss/train", avg_l_epoch, epoch)
            if writer: writer.add_scalar("IoU/train",  avg_i_epoch, epoch)
            print(f"*** Epoch {epoch+1} done: loss {avg_l_epoch:.4f}, IoU {avg_i_epoch:.4f}", flush=True)
        print(f"DEBUG: Epoch {epoch+1}/{args.epochs} - Finished.", flush=True)

    # ── Finalize ────────────────────────────────────────────────────────
    print("DEBUG: Training loop finished. Finalizing...", flush=True)
    if rank==0:
        print(f"DEBUG: Rank 0 saving model to {args.output} and CSV to {args.csv_output}...", flush=True)
        torch.save(sam.module.state_dict(), args.output)
        with open(args.csv_output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class_id","mean_iou","std_iou","count"])
            for cls, ious in final_ious.items(): # Original variable name 'ious'
                if ious: # Check if the list of ious is not empty
                    w.writerow([cls, np.mean(ious), np.std(ious), len(ious)])
                else: # Handle case for classes with no IoUs recorded to prevent np.mean/std errors
                    w.writerow([cls, 0.0, 0.0, 0]) # Provide sensible defaults
        if writer: writer.close()
        print("DEBUG: Rank 0 model and CSV saved. Writer closed.", flush=True)

    print("DEBUG: Destroying DDP process group...", flush=True)
    dist.destroy_process_group()
    print("DEBUG: DDP process group destroyed. main() finished.", flush=True)

if __name__ == "__main__":
    print("DEBUG: Script execution started (__name__ == '__main__').", flush=True)
    main()