"""风格 backbone 的对比学习预训练（SupCon）。

预训练目标：让 StyleBackbone 学到"风格不变性 + 内容判别性"的表示：
  - 正样本：同作者不同字 + 同图自增强对（多视图 SupCon）
  - 负样本：batch 内非同作者样本，包含同字不同作者的硬负样本

数据：data/StyleDataset/Train/{author}/{unicode}.jpg
输出：checkpoints/style_pretrain/style_backbone_ep{N}.pt
       结构：{'backbone': state_dict, 'proj_head': state_dict, 'epoch': N, 'args': ...}
       下游可经 train.py --style_pretrained 路径加载。

用法：
  python pretrain_style.py --epochs 30 --k_authors 16 --m_chars 8 --hold_out 24
  python pretrain_style.py --epochs 2  --n_batches_per_epoch 20 --hold_out 8    # 冒烟

参考：Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
"""

import argparse
import math
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.amp import autocast, GradScaler

from datareader import _resize_pad
from model import StyleBackbone


# ============================================================================
# 数据集 / Sampler
# ============================================================================


class WriterContrastiveDataset(Dataset):
    """扫描所有作者目录，建立 (author_id, char_id, file) 索引。

    返回:
        image:     [1, S, S] float32 in [-1, 1]
        author_id: int  (0..n_authors-1)
        char_id:   int  (0..n_chars-1)
    """

    def __init__(self, root: str = 'data/StyleDataset', split: str = 'train',
                 size: int = 64, hold_out_authors: int = 0, seed: int = 42):
        assert split in ('train', 'val')
        self.size = size
        train_root = Path(root) / 'Train'
        authors_all = sorted([p.name for p in train_root.iterdir() if p.is_dir()])

        # 用固定 seed 划分 hold-out 作者，保证 train/val 一致
        rng = random.Random(seed)
        rng.shuffle(authors_all)
        if hold_out_authors > 0:
            val_authors = set(authors_all[:hold_out_authors])
            train_authors = set(authors_all[hold_out_authors:])
        else:
            val_authors, train_authors = set(), set(authors_all)
        active = train_authors if split == 'train' else val_authors

        self.authors = sorted(active)
        self.author_to_idx = {a: i for i, a in enumerate(self.authors)}

        self.items: List[Tuple[int, int, Path]] = []
        self.char_to_idx: dict = {}
        for author in self.authors:
            adir = train_root / author
            for jpg in adir.glob('*.jpg'):
                uni = jpg.stem.upper()
                if uni not in self.char_to_idx:
                    self.char_to_idx[uni] = len(self.char_to_idx)
                self.items.append((self.author_to_idx[author],
                                   self.char_to_idx[uni], jpg))
        self.by_author: dict = {}
        for i, (a, _, _) in enumerate(self.items):
            self.by_author.setdefault(a, []).append(i)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        author_id, char_id, path = self.items[idx]
        arr = _resize_pad(Image.open(path).convert('L'), self.size)
        return arr[None], author_id, char_id


class BalancedAuthorBatchSampler(Sampler):
    """每 batch 抽 K 个作者，每作者抽 M 个不同字符 → batch_size = K*M。

    同作者多字 → 自然形成正样本对；不同作者 → 负样本对（含同字硬负样本）。
    """

    def __init__(self, dataset: WriterContrastiveDataset,
                 k_authors: int = 16, m_chars: int = 8,
                 n_batches: int = None, seed: int = 42):
        self.by_author = dataset.by_author
        self.authors = list(self.by_author.keys())
        self.k = k_authors
        self.m = m_chars
        self.n_batches = n_batches or max(1, len(dataset) // (k_authors * m_chars))
        self.rng = random.Random(seed)

    def __iter__(self):
        for _ in range(self.n_batches):
            picked = self.rng.sample(self.authors, self.k)
            batch = []
            for a in picked:
                pool = self.by_author[a]
                if len(pool) >= self.m:
                    sel = self.rng.sample(pool, self.m)
                else:
                    sel = self.rng.choices(pool, k=self.m)
                batch.extend(sel)
            yield batch

    def __len__(self) -> int:
        return self.n_batches


# ============================================================================
# 增强 / 模型 / 损失
# ============================================================================


def style_augment(x: torch.Tensor, training: bool = True) -> torch.Tensor:
    """温和增强：小幅仿射 + 加性噪声。汉字方向有意义，不做翻转/大角度旋转。"""
    if not training:
        return x
    B = x.size(0)
    angle = (torch.rand(B, device=x.device) * 6 - 3)             # ±3°
    tx = (torch.rand(B, device=x.device) * 4 - 2) / x.size(-1)    # ±2px
    ty = (torch.rand(B, device=x.device) * 4 - 2) / x.size(-2)
    scale = 0.95 + torch.rand(B, device=x.device) * 0.10          # 0.95~1.05

    theta = torch.zeros(B, 2, 3, device=x.device)
    cos = torch.cos(angle * math.pi / 180)
    sin = torch.sin(angle * math.pi / 180)
    theta[:, 0, 0] = cos / scale
    theta[:, 0, 1] = -sin / scale
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin / scale
    theta[:, 1, 1] = cos / scale
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, x.shape, align_corners=False)
    # 背景白色 (+1)，affine 后边缘填充用 border 复制最外像素
    x_pad = F.grid_sample(x, grid, mode='bilinear',
                          padding_mode='border', align_corners=False)
    noise = torch.randn_like(x_pad) * 0.02
    return (x_pad + noise).clamp(-1, 1)


class ContrastiveStyleModel(nn.Module):
    """backbone + global pool + projection head。projection head 仅供 SupCon loss 使用。"""

    def __init__(self, c_mid: int = 64, c_deep: int = 96, proj_dim: int = 128):
        super().__init__()
        self.backbone = StyleBackbone(c_mid=c_mid, c_deep=c_deep)
        self.proj_head = nn.Sequential(
            nn.Linear(c_deep, c_deep),
            nn.ReLU(inplace=True),
            nn.Linear(c_deep, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, f_deep = self.backbone(x)
        g = F.adaptive_avg_pool2d(f_deep, 1).flatten(1)        # [B, c_deep]
        z = self.proj_head(g)
        return F.normalize(z, dim=-1)                          # [B, proj_dim]


def supcon_loss(features: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.1) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al. 2020)。

    features: [N, D]（假定已 L2-normalized）
    labels:   [N]  每个样本的 author_id
    """
    device = features.device
    N = features.size(0)

    logits = features @ features.T / temperature
    logits_max, _ = logits.max(dim=1, keepdim=True)
    logits = logits - logits_max.detach()                       # 数值稳定

    labels = labels.view(-1, 1)
    pos_mask = torch.eq(labels, labels.T).float().to(device)
    self_mask = torch.eye(N, device=device).bool()
    pos_mask.masked_fill_(self_mask, 0.0)

    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(dim=1).clamp_min(1.0)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count
    return -mean_log_prob_pos.mean()


# ============================================================================
# 训练
# ============================================================================


@torch.no_grad()
def evaluate_retrieval(model, val_loader, device) -> float:
    """简易 retrieval accuracy：对每个样本，最近邻是否同 author。"""
    model.eval()
    feats, labels = [], []
    for imgs, aids, _ in val_loader:
        imgs = imgs.to(device).float()
        z = model(imgs)
        feats.append(z.cpu())
        labels.append(aids)
    feats = torch.cat(feats, 0)
    labels = torch.cat(labels, 0)
    sim = feats @ feats.T
    sim.fill_diagonal_(-1.0)
    nn_idx = sim.argmax(dim=1)
    acc = (labels[nn_idx] == labels).float().mean().item()
    model.train()
    return acc



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/StyleDataset')
    p.add_argument('--out_dir', default='checkpoints/style_pretrain')
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--c_mid', type=int, default=96,
                   help='backbone c_mid，需与 train.py 的 c_style_mid 一致')
    p.add_argument('--c_deep', type=int, default=160,
                   help='backbone c_deep，需与 train.py 的 c_style_deep 一致')
    p.add_argument('--proj_dim', type=int, default=128)
    p.add_argument('--k_authors', type=int, default=16)
    p.add_argument('--m_chars', type=int, default=8)
    p.add_argument('--n_batches_per_epoch', type=int, default=500)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--hold_out', type=int, default=24)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    amp = (not args.no_amp) and device.type == 'cuda'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = WriterContrastiveDataset(args.root, 'train', args.size,
                                        hold_out_authors=args.hold_out, seed=args.seed)
    val_ds = WriterContrastiveDataset(args.root, 'val', args.size,
                                      hold_out_authors=args.hold_out, seed=args.seed)
    sampler = BalancedAuthorBatchSampler(
        train_ds, args.k_authors, args.m_chars,
        n_batches=args.n_batches_per_epoch, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f'[data] train {len(train_ds)} samples / {len(train_ds.authors)} authors')
    print(f'[data] val   {len(val_ds)} samples / {len(val_ds.authors)} authors (unseen)')

    model = ContrastiveStyleModel(args.c_mid, args.c_deep, args.proj_dim).to(device)
    n_params = sum(p.numel() for p in model.backbone.parameters())
    print(f'[model] backbone c_mid={args.c_mid} c_deep={args.c_deep} '
          f'params={n_params/1e6:.2f}M  device={device}  amp={amp}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda', enabled=amp)

    for ep in range(args.epochs):
        model.train()
        t0, ep_loss = time.time(), 0.0
        for imgs, aids, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True).float()
            aids = aids.to(device, non_blocking=True).long()

            v1 = style_augment(imgs)
            v2 = style_augment(imgs)
            x = torch.cat([v1, v2], dim=0)
            y = torch.cat([aids, aids], dim=0)

            with autocast('cuda', enabled=amp):
                z = model(x)
                loss = supcon_loss(z, y, args.temperature)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ep_loss += loss.item()

        scheduler.step()
        avg_loss = ep_loss / max(1, len(train_loader))
        dt = time.time() - t0
        msg = (f'[ep {ep+1}/{args.epochs}] loss={avg_loss:.4f} '
               f'lr={scheduler.get_last_lr()[0]:.2e} time={dt:.1f}s')

        if len(val_ds) > 0 and (ep + 1) % 2 == 0:
            acc = evaluate_retrieval(model, val_loader, device)
            msg += f' val_retr_acc={acc:.4f}'
        print(msg)

        if (ep + 1) % args.save_every == 0 or (ep + 1) == args.epochs:
            ckpt = {
                'backbone': model.backbone.state_dict(),
                'proj_head': model.proj_head.state_dict(),
                'epoch': ep + 1,
                'args': vars(args),
            }
            out_path = out_dir / f'style_backbone_ep{ep+1}.pt'
            torch.save(ckpt, out_path)
            print(f'  saved -> {out_path}')


if __name__ == '__main__':
    main()
