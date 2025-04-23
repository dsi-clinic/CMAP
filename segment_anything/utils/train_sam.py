#!/usr/bin/env python
"""
train_sam.py
Distributed Data Parallel (DDP) version for multi-GPU training on a single node.
Usage example (with torch.distributed.run):
  srun python -m torch.distributed.run \
    --nproc_per_node=$SLURM_NTASKS train_sam.py \
    --image-dir /path/to/images \
    --mask-dir  /path/to/masks \
    --checkpoint /path/to/sam_vit_h.pth \
    --output fine_tuned_ddp.pth \
    --epochs 5 --batch-size 1 --accum-steps 4 --lr 1e-4
"""
import sys, os, argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
# adjust this path to where your repo lives
sys.path.append("/home/gregoryc25/CMAP/segment_anything_source_code")
from segment_anything.build_sam import sam_model_registry
from segment_anything.predictor import SamPredictor

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune SAM mask decoder with DDP")
    p.add_argument("--image-dir",   required=True, help="Path to input images directory")
    p.add_argument("--mask-dir",    required=True, help="Path to binary masks directory")
    p.add_argument("--checkpoint",  required=True, help="Path to sam_vit_h.pth checkpoint")
    p.add_argument("--output",      default="fine_tuned_ddp.pth", help="Where to save fine-tuned weights (rank 0 only)")
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--batch-size",  type=int,   default=1)
    p.add_argument("--accum-steps", type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--local_rank",  type=int,   default=int(os.environ.get('LOCAL_RANK', 0)))
    return p.parse_args()

class AerialDataset(Dataset):
    def __init__(self, image_dir, mask_dir, mask_prefix="mask_"):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        imgs = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png','.jpg','.tif'))])
        self.pairs = []
        for img in imgs:
            m = mask_prefix + img
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
        msk_path = os.path.join(self.mask_dir,  mask_fn)
        img = Image.open(img_path).convert('RGB')
        msk = Image.open(msk_path).convert('L')
        w,h = img.size
        r = min(1024/w, 1024/h)
        if r < 1.0:
            img = img.resize((int(w*r), int(h*r)), Image.BILINEAR)
            msk = msk.resize((int(w*r), int(h*r)), Image.NEAREST)
        img_np = np.array(img)
        msk_np = (np.array(msk) > 0).astype(np.uint8)
        kernel = np.ones((5,5), np.uint8)
        er = cv2.erode(msk_np, kernel, iterations=1)
        if er.max() == 0:
            er = msk_np
        coords = np.argwhere(er > 0)
        if coords.size:
            yi, xi = coords[np.random.randint(len(coords))]
            point = np.array([int(xi), int(yi)], np.int32)
        else:
            point = np.array([0,0], np.int32)
        return img_np, msk_np, point


def sam_collate_fn(batch):
    imgs, msks, pts = zip(*batch)
    return list(imgs), list(msks), list(pts)


def main():
    args = parse_args()

    # DDP init
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    rank = dist.get_rank()

    if rank == 0:
        print(f"Using device(s): {torch.cuda.device_count()} GPUs for DDP")

    # 1) load model
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    sam = sam_model_registry['vit_h'](checkpoint=str(ckpt))
    sam.to(device)
    predictor = SamPredictor(sam)

    # 2) freeze encoders
    for p in sam.image_encoder.parameters():   p.requires_grad = False
    for p in sam.prompt_encoder.parameters():  p.requires_grad = False
    sam.image_encoder.eval(); sam.prompt_encoder.eval(); sam.mask_decoder.train()

    # 3) wrap with DDP
    sam = DDP(sam, device_ids=[args.local_rank], output_device=args.local_rank)

    # 4) dataset + sampler + loader
    ds = AerialDataset(args.image_dir, args.mask_dir)
    sampler = DistributedSampler(ds)
    loader  = DataLoader(ds,
                         batch_size=args.batch_size,
                         sampler=sampler,
                         num_workers=4,
                         collate_fn=sam_collate_fn)

    # 5) optimizer & scaler
    optimizer = torch.optim.AdamW(sam.module.mask_decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler    = torch.cuda.amp.GradScaler()

    # 6) training loop
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        running_loss = running_iou = 0.0
        total = len(loader)
        for i, (imgs, msks, pts) in enumerate(loader, 1):
            img_np, gt_np, point = imgs[0], msks[0], pts[0]
            predictor.model = sam.module  # ensure predictor uses the wrapped model
            predictor.set_image(img_np)
            emb = predictor.get_image_embedding().to(device)
            ipt = np.expand_dims(point,0).astype(np.int32)
            lbl = np.array([[1]],np.int32)
            pt_t = torch.from_numpy(ipt).to(device).unsqueeze(0)
            lb_t = torch.from_numpy(lbl).to(device)
            sparse, dense = sam.module.prompt_encoder((pt_t,lb_t), None, None)

            with torch.cuda.amp.autocast():
                low_res, _ = sam.module.mask_decoder(
                    image_embeddings=emb,
                    image_pe=sam.module.prompt_encoder.get_dense_pe().to(device),
                    sparse_prompt_embeddings=sparse.to(device),
                    dense_prompt_embeddings=dense.to(device),
                    multimask_output=True)
                lr = low_res[:,0:1]
                H,W = gt_np.shape
                up = torch.nn.functional.interpolate(lr, size=(H,W), mode='bilinear', align_corners=False)
                gt_t = torch.from_numpy(gt_np.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(up, gt_t)

            scaler.scale(loss).backward()
            if i % args.accum_steps == 0 or i == total:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

            running_loss += loss.item()
            with torch.no_grad():
                pred = (torch.sigmoid(up)>0.5).float()
                inter = (pred*gt_t).sum().item()
                union= (pred+gt_t - pred*gt_t).sum().item()
                running_iou += (inter/union) if union else 0

            if i % 10 == 0 and rank == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Step {i}/{total} "
                      f"– Loss {running_loss/i:.4f}, IoU {running_iou/i:.4f}")

        if rank == 0:
            print(f"Epoch {epoch+1} complete – Avg Loss {running_loss/total:.4f}, Avg IoU {running_iou/total:.4f}\n")

    # 7) save only on rank 0
    if rank == 0:
        torch.save(sam.module.state_dict(), args.output)
        print(f"Fine-tuned weights saved to {args.output}")

    dist.destroy_process_group()

if __name__=='__main__':
    main()
