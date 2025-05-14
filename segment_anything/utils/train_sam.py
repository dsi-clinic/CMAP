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
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

# TorchGeo
from torchgeo.datasets import NAIP, stack_samples
from torchgeo.samplers import RandomGeoSampler, Units

# Local dataset subclass
from prompted_kc import PromptedKaneCounty
from data.kc import KaneCounty

# SAM
sys.path.append(str(repo_root / "segment_anything_source_code"))
from segment_anything.build_sam import sam_model_registry
from segment_anything.predictor import SamPredictor

def parse_args():
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
    p.add_argument("--shape-path", required=True, help="Path to KC .gpkg file")
    p.add_argument("--layer-name", default="Basins", help="Layer name in GeoPackage")
    p.add_argument("--chip-size",  type=int, default=512, help="Patch size in pixels")
    # DDP
    p.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK",0)))
    return p.parse_args()

def main():
    args = parse_args()

    # ── DDP init ────────────────────────────────────────────────────────
    os.environ.setdefault("MASTER_ADDR","localhost")
    os.environ.setdefault("MASTER_PORT", str(10000 + random.randrange(10000)))
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda:{args.local_rank}")
    rank  = dist.get_rank()
    world = dist.get_world_size()

    # ── Logging / TB ────────────────────────────────────────────────────
    writer = SummaryWriter("runs/sam_kc") if rank == 0 else None
    debug_dir = Path("debug_images")
    if rank == 0:
        debug_dir.mkdir(exist_ok=True)

    # ── Load & freeze SAM ───────────────────────────────────────────────
    sam = sam_model_registry["vit_h"](checkpoint=str(args.checkpoint))
    sam.to(device)
    for p in sam.image_encoder.parameters(): p.requires_grad = False
    for p in sam.prompt_encoder.parameters(): p.requires_grad = False
    sam.image_encoder.eval(); sam.prompt_encoder.eval(); sam.mask_decoder.train()
    sam = DDP(sam, device_ids=[args.local_rank], output_device=args.local_rank)

    # ── Build TorchGeo dataset & sampler ────────────────────────────────
    naip_ds = NAIP(args.naip_root)
    # identity map: GT id → same id
    idmap = {k: k for k in KaneCounty.all_labels.keys()}
    label_ds = PromptedKaneCounty(
        args.shape_path,
        (args.layer_name, idmap, args.chip_size, naip_ds.crs, naip_ds.res)
    )
    combined = naip_ds & label_ds

    sampler = RandomGeoSampler(
        combined,
        size=args.chip_size,
        length=max(1, len(combined)//world),
        units=Units.PIXELS
    )
    loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=stack_samples,
        pin_memory=True,
    )

    # ── Optimizer & scaler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(sam.module.mask_decoder.parameters(), lr=args.lr)
    scaler    = torch.cuda.amp.GradScaler()

    final_ious = defaultdict(list)

    # ── Training loop ───────────────────────────────────────────────────
    for epoch in range(args.epochs):
        sam.module.mask_decoder.train()
        running_loss = 0.0
        running_iou  = 0.0
        steps = 0

        for batch in loader:
            imgs   = batch["image"].to(device)  # (B,C,H,W)
            masks  = batch["mask"].to(device)   # (B,H,W)
            points = batch["point"]             # list of [x,y]
            B = imgs.size(0)

            loss_batch = 0.0
            with torch.cuda.amp.autocast():
                for b in range(B):
                    # 1) embed
                    img_np = imgs[b].permute(1,2,0).cpu().numpy().astype(np.uint8)
                    predictor = SamPredictor(sam.module)
                    predictor.set_image(img_np)
                    emb = predictor.get_image_embedding().to(device)
                    # 2) prompt
                    pt = torch.from_numpy(points[b]).to(device).unsqueeze(0).unsqueeze(0).float()
                    lbl = torch.ones((1,1), device=device)
                    sp, dn = sam.module.prompt_encoder((pt,lbl), None, None)
                    # 3) decode
                    low, _ = sam.module.mask_decoder(
                        image_embeddings=emb,
                        image_pe=sam.module.prompt_encoder.get_dense_pe().to(device),
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
                    cls = int(masks[b][points[b][1], points[b][0]].item())
                    final_ious[cls].append(iou)
                    running_iou += iou

                # normalize by accumulation
                loss_batch = loss_batch / (args.accum_steps * B)

            scaler.scale(loss_batch).backward()
            steps += 1

            if steps % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss_batch.item()

            if rank==0 and steps%10==0:
                avg_l = running_loss/steps
                avg_i = running_iou/(steps*B)
                print(f"[Epoch {epoch+1}] Step {steps} → loss {avg_l:.4f}, IoU {avg_i:.4f}")

        dist.barrier()
        if rank==0:
            avg_l = running_loss/steps
            avg_i = running_iou/(steps*B)
            writer.add_scalar("Loss/train", avg_l, epoch)
            writer.add_scalar("IoU/train",  avg_i, epoch)
            print(f"*** Epoch {epoch+1} done: loss {avg_l:.4f}, IoU {avg_i:.4f}")

    # ── Finalize ────────────────────────────────────────────────────────
    if rank==0:
        torch.save(sam.module.state_dict(), args.output)
        with open(args.csv_output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class_id","mean_iou","std_iou","count"])
            for cls, ious in final_ious.items():
                w.writerow([cls, np.mean(ious), np.std(ious), len(ious)])
        writer.close()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
