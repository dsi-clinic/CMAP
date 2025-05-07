#!/usr/bin/env python
"""
train_sam.py
Distributed Data Parallel (DDP) version for multi-GPU training on a single node.
"""
import sys, os, argparse, random, csv
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

sys.path.append("/home/gregoryc25/CMAP/segment_anything_source_code")
from segment_anything.build_sam import sam_model_registry
from segment_anything.predictor import SamPredictor

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune SAM mask decoder with DDP")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--mask-prefix", default="mask_")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="fine_tuned_ddp.pth")
    p.add_argument("--csv-output", default="per_class_ious_ddp.csv")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--local_rank", type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    return p.parse_args()

class AerialDataset(Dataset):
    def __init__(self, image_dir, mask_dir, mask_prefix="mask_"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_prefix = mask_prefix
        imgs = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.tif'))])
        self.pairs = []
        for img in imgs:
            m = self.mask_prefix + img
            if os.path.exists(os.path.join(mask_dir, m)):
                self.pairs.append((img, m))
            else:
                print(f"Warning: mask not found for {img}")
        if dist.get_rank() == 0:
            print(f"Dataset: {len(self.pairs)} image-mask pairs found.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_fn, mask_fn = self.pairs[idx]
        img_path = os.path.join(self.image_dir, img_fn)
        msk_path = os.path.join(self.mask_dir, mask_fn)
        img = Image.open(img_path).convert('RGB')
        msk = Image.open(msk_path).convert('L')
        w, h = img.size
        r = min(1024 / w, 1024 / h)
        if r < 1.0:
            img = img.resize((int(w * r), int(h * r)), Image.BILINEAR)
            msk = msk.resize((int(w * r), int(h * r)), Image.NEAREST)
        img_np = np.array(img)
        msk_np = np.array(msk)

        valid_labels = np.unique(msk_np)
        valid_labels = valid_labels[valid_labels > 0]
        if len(valid_labels) > 0:
            selected_cls = np.random.choice(valid_labels)
            coords = np.argwhere(msk_np == selected_cls)
            yi, xi = coords[np.random.randint(len(coords))]
            point = np.array([int(xi), int(yi)], np.int32)
        else:
            selected_cls = 1
            point = np.array([0, 0], np.int32)

        return img_np, msk_np, point

def sam_collate_fn(batch):
    imgs, msks, pts = zip(*batch)
    return list(imgs), list(msks), list(pts)

def compute_iou_per_class(pred, target, class_ids=(0, 1)):
    ious = {}
    for cls in class_ids:
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        ious[cls] = intersection / union if union != 0 else 0.0
    return ious

def main():
    args = parse_args()

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(random.randint(10000, 20000))

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    rank = dist.get_rank()

    writer = SummaryWriter(log_dir="runs/sam_finetune") if rank == 0 else None

    if rank == 0:
        print(f"Using device(s): {torch.cuda.device_count()} GPUs for DDP")

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    sam = sam_model_registry['vit_h'](checkpoint=str(ckpt))
    sam.to(device)
    predictor = SamPredictor(sam)

    for p in sam.image_encoder.parameters(): p.requires_grad = False
    for p in sam.prompt_encoder.parameters(): p.requires_grad = False
    sam.image_encoder.eval(); sam.prompt_encoder.eval(); sam.mask_decoder.train()

    sam = DDP(sam, device_ids=[args.local_rank], output_device=args.local_rank)
    ds = AerialDataset(args.image_dir, args.mask_dir, mask_prefix=args.mask_prefix)
    sampler = DistributedSampler(ds)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=4, collate_fn=sam_collate_fn)

    optimizer = torch.optim.AdamW(sam.module.mask_decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()

    final_class_ious = defaultdict(list)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        running_loss = running_iou = 0.0
        total = len(loader)
        class_iou_totals = defaultdict(float)
        class_counts = defaultdict(int)

        for i, (imgs, msks, pts) in enumerate(loader, 1):
            img_np, gt_np, point = imgs[0], msks[0], pts[0]
            predictor.model = sam.module
            predictor.set_image(img_np)
            emb = predictor.get_image_embedding().to(device)
            ipt = np.expand_dims(point, 0).astype(np.int32)
            lbl = np.array([[1]], np.int32)
            pt_t = torch.from_numpy(ipt).to(device).unsqueeze(0)
            lb_t = torch.from_numpy(lbl).to(device)
            sparse, dense = sam.module.prompt_encoder((pt_t, lb_t), None, None)

            with torch.amp.autocast(device_type='cuda'):
                low_res, _ = sam.module.mask_decoder(
                    image_embeddings=emb,
                    image_pe=sam.module.prompt_encoder.get_dense_pe().to(device),
                    sparse_prompt_embeddings=sparse.to(device),
                    dense_prompt_embeddings=dense.to(device),
                    multimask_output=True)
                lr = low_res[:, 0:1]
                H, W = gt_np.shape
                up = torch.nn.functional.interpolate(lr, size=(H, W), mode='bilinear', align_corners=False)
                gt_t = torch.from_numpy(gt_np.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(up, gt_t)

            scaler.scale(loss).backward()
            if i % args.accum_steps == 0 or i == total:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

            running_loss += loss.item()
            with torch.no_grad():
                pred = (torch.sigmoid(up) > 0.5).float()
                pred_mask = pred.squeeze().cpu().int()
                gt_mask = gt_t.squeeze().cpu().int()
                class_ids = torch.unique(gt_mask).tolist()
                ious = compute_iou_per_class(pred_mask, gt_mask, class_ids)
                for cls, iou in ious.items():
                    class_iou_totals[cls] += iou
                    class_counts[cls] += 1
                    final_class_ious[cls].append(iou)
                inter = (pred * gt_t).sum().item()
                union = (pred + gt_t - pred * gt_t).sum().item()
                running_iou += (inter / union) if union else 0

            if i % 10 == 0 and rank == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Step {i}/{total} "
                      f"– Loss {running_loss/i:.4f}, IoU {running_iou/i:.4f}")

        if rank == 0:
            print(f"Epoch {epoch+1} complete – Avg Loss {running_loss/total:.4f}, Avg IoU {running_iou/total:.4f}\n")
            writer.add_scalar("Loss/train", running_loss / total, epoch)
            writer.add_scalar("IoU/train", running_iou / total, epoch)
            for cls in sorted(class_iou_totals):
                avg_cls_iou = class_iou_totals[cls] / class_counts[cls]
                writer.add_scalar(f"IoU/Class_{cls}", avg_cls_iou, epoch)

    if rank == 0:
        torch.save(sam.module.state_dict(), args.output)
        print(f"Fine-tuned weights saved to {args.output}")
        writer.close()

        # Save per-class IOU CSV
        with open(args.csv_output, 'w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["class_id", "mean_iou", "std_iou", "count"])
            for cls, iou_list in final_class_ious.items():
                writer_csv.writerow([cls, np.mean(iou_list), np.std(iou_list), len(iou_list)])
            print(f"Saved per-class IoUs to {args.csv_output}")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
