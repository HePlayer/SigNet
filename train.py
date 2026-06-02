"""训练入口：支持 P0 自重建和 P1 风格迁移。

P0 自重建（默认）：
    python train.py --mode p0 --author 1126-f --epochs 30 --batch 32
P1 风格迁移（K=印刷体, V=手写体）：
    python train.py --mode p1 --author 1126-f --epochs 30 --batch 32
快速冒烟：
    python train.py --mode p1 --smoke

标准 CTM 默认保守超参：neurons=128, T=6, M=4, sync_pairs=64。
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from datareader import MiniReconDataset, collate, StylePairDataset, collate_pair
from losses import (SobelEdge, compute_losses, diffusion_losses,
                    lsgan_d_loss, lsgan_g_loss)
from model import (CTMRecon, CTMHybridRecon, PatchDiscriminator,
                   RecurrentDiffusionRecon)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, default='p1', choices=['p0', 'p1'],
                   help='p0=自重建, p1=风格迁移(K=印刷体,V=手写体)')
    p.add_argument('--root', type=str, default='data/StyleDataset')
    p.add_argument('--struct_root', type=str, default='data/StructDataset',
                   help='P1 模式：印刷体图像目录')
    p.add_argument('--author', type=str, default='1126-f')
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--T', type=int, default=16, help='CTM internal ticks')
    p.add_argument('--k_top', type=int, default=1,
                   help='CTM loss 选取的 top-K min-loss tick 数（per-sample），'
                        '另加 1 个 max-certainty tick；K=1 等价 CTM 论文原版')
    p.add_argument('--reveal_mode', type=str, default='tau',
                   choices=['tau', 'center_out', 'edge_in'],
                   help='Spatial Progressive Reveal 模式：tau 基于连通分量+线性面积增长'
                        '（默认）；center_out / edge_in 为几何 ablation')
    p.add_argument('--reveal_soft', type=float, default=0.05,
                   help='huu mask 软边界宽度：tau 模式下为 fraction-of-progress'
                        '（建议 0.03~0.08），几何模式下为像素（建议 1~3）')
    p.add_argument('--gamma', type=float, default=0.9,
                   help='加权深度监督衰减系数：w_t = γ^(T-1-t)，γ→1 等权重，γ→0 仅最后')
    p.add_argument('--k_win', type=int, default=2,
                   help='滚动窗口监督：tick t 仅监督 (M_t - M_{t-k_win}) 最近 k_win 步'
                        '新揭露的 ink；前 k_win 个 tick 作 warmup 使用累计 M_t；'
                        '最后 1 个 tick 始终用完整 target。<=0 时退回累计揭露（旧行为）。')
    p.add_argument('--M', type=int, default=4, help='NLM pre-activation history length')
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--neurons', type=int, default=128,
                   help='P0 标准 CTM 内部神经元数')
    p.add_argument('--sync_pairs', type=int, default=64,
                   help='固定随机采样的同步 neuron pair 数量')
    p.add_argument('--attn_dim', type=int, default=64,
                   help='P0 标准 CTM 的 attention 维度')
    # ---- P1 空间化 CTM 专用 ----
    p.add_argument('--authors', type=str,
                   default='1001-f,1002-f,1003-f,1004-f,1005-f,1006-f',
                   help='P1 多作者训练：逗号分隔的作者列表（默认 6 作者）；'
                        '为空字符串则回退到 --author 单作者')
    p.add_argument('--n_refs', type=int, default=8,
                   help='P1 每个样本采样多少张同作者风格参考字')
    p.add_argument('--c_z', type=int, default=192,
                   help='P1 空间化 CTM 每个位置的神经元数')
    p.add_argument('--sp_sync_pairs', type=int, default=80,
                   help='P1 空间化 CTM 同步 neuron pair 数')
    p.add_argument('--sp_attn_dim', type=int, default=128,
                   help='P1 空间化 CTM 的 attention 维度')
    p.add_argument('--c_style_mid', type=int, default=96)
    p.add_argument('--c_style_deep', type=int, default=160)
    p.add_argument('--style_encoder', type=str, default='backbone',
                   choices=['shallow', 'backbone'],
                   help='风格编码器类型：shallow=原 StyleEncoder, backbone=更深的 StyleBackbone')
    p.add_argument('--style_pretrained', type=str, default='',
                   help='对比预训练 backbone 权重路径（仅 style_encoder=backbone 时生效）')
    p.add_argument('--freeze_style_epochs', type=int, default=30,
                   help='前 N 个 epoch 冻结 style backbone 参数；N=0 不冻结')
    p.add_argument('--c_content_aux', type=int, default=64,
                   help='P1 内容编码器辅助层通道数（输出 32x32）')
    p.add_argument('--c_content_main', type=int, default=128,
                   help='P1 内容编码器主层通道数（输出 16x16）')
    p.add_argument('--r_init', type=float, default=-3.0,
                   help='r_raw 初始值；decay≈exp(-softplus(r))，越小记忆越长')
    # ---- P1 对抗训练（Step 1：PatchGAN + LSGAN） ----
    p.add_argument('--w_adv', type=float, default=1.0,
                   help='P1 对抗损失权重（仅作用于最后一个 tick 的输出）')
    p.add_argument('--w_perceptual', type=float, default=1.0,
                   help='P1 感知损失权重（VGG19 relu2_2+relu3_3，仅作用于最后一个 tick）')
    p.add_argument('--dist_lambda', type=float, default=0.05,
                   help='Content 注意力空间距离惩罚系数（0 表示无惩罚）')
    p.add_argument('--lr_d', type=float, default=2e-4,
                   help='判别器学习率')
    p.add_argument('--d_base', type=int, default=32,
                   help='PatchDiscriminator 基础通道数')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--out_dir', type=str, default='runs/p1')
    p.add_argument('--vis_every', type=int, default=2, help='每多少 epoch 保存一次可视化')
    p.add_argument('--smoke', action='store_true', help='极少样本快速冒烟')
    p.add_argument('--max_train_items', type=int, default=-1,
                   help='截断训练集到前 N 条（-1 不限）；用于容量/欠拟合诊断，val 集不受影响')
    p.add_argument('--model_type', type=str, default='hybrid',
                   choices=['standard', 'hybrid'],
                   help='standard=原CTMRecon, '
                        'hybrid=新CTMHybridRecon(Q融合+风格放大+PixelShuffle)')
    p.add_argument('--style_amp', type=float, default=2.0,
                   help='CTMHybridRecon 风格放大系数')
    # ---- 循环去噪重建（Recurrent Diffusion）----
    p.add_argument('--diffusion', action='store_true',
                   help='启用 RecurrentDiffusionRecon：像素空间 x0-预测扩散，'
                        '外层采样循环=循环深度；与 CTM 分支互斥')
    p.add_argument('--diff_inner', type=int, default=1,
                   help='DenoiseCore 每个去噪步内部残差更新次数 T_inner')
    p.add_argument('--diff_w_pix', type=float, default=1.0,
                   help='扩散 x0 的 MSE 权重')
    p.add_argument('--diff_w_edge', type=float, default=0.5,
                   help='扩散 Sobel 边缘 L1 权重')
    p.add_argument('--diff_fg_w', type=float, default=2.0,
                   help='扩散 x0 MSE 的前景(ink)加权系数；前景/背景已各按面积'
                        '归一化，1.0=前景背景等权，>1 进一步强调笔画以抑制缺笔')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """[-1,1] tensor → [0,255] uint8 ndarray，单张 [H,W]。"""
    t = t.detach().clamp(-1, 1).cpu()
    arr = ((t + 1.0) * 127.5).round().byte().numpy()
    return arr


def save_grid(outs, target, path: Path, max_n: int = 6, extras=None):
    """保存可视化网格到 path（PNG）。

    行布局：
    - P0：[target | tick_1 | tick_2 | ...]
    - P1（extras=[x_c, x_s]）：[x_c | x_s | x_gt | tick_1 | ... ]

    extras 中 x_s 形状可能为 [B,1,H,W] 或 [B,N,1,H,W]，后者只显示第一张参考。
    """
    B = min(max_n, target.size(0))
    rows = []
    if extras:
        for ex in extras:
            if ex.dim() == 5:
                ex = ex[:, 0]
            rows.append(np.concatenate([to_uint8(ex[i, 0]) for i in range(B)], axis=1))
    rows.append(np.concatenate([to_uint8(target[i, 0]) for i in range(B)], axis=1))
    for o in outs:
        rows.append(np.concatenate([to_uint8(o[i, 0]) for i in range(B)], axis=1))
    grid = np.concatenate(rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path)


@torch.no_grad()
def evaluate_p0(model, loader, sobel, device, k_top=1,
                reveal_mode='tau', reveal_soft=0.05, gamma=0.9,
                perceptual_loss=None, w_perceptual=0.0, k_win=2):
    """P0 验证：单输入自重建。"""
    model.eval()
    tot_pix, tot_edge, tot_perc, tot_last, n = 0.0, 0.0, 0.0, 0.0, 0
    last_outs, last_x = None, None
    for x, _ in loader:
        x = x.to(device)
        with autocast('cuda'):
            outs = model(x)
            ld = compute_losses(outs, x, sobel, k_top=k_top,
                                reveal_mode=reveal_mode, reveal_soft=reveal_soft,
                                gamma=gamma,
                                perceptual_loss=perceptual_loss,
                                w_perceptual=w_perceptual,
                                k_win=k_win)
        bs = x.size(0)
        tot_pix += ld['pix'].item() * bs
        tot_edge += ld['edge'].item() * bs
        tot_perc += ld['perceptual'].item() * bs
        tot_last += torch.nn.functional.l1_loss(
            outs[-1].float(), x.float()).item() * bs
        n += bs
        last_outs, last_x = outs, x
    model.train()
    return tot_pix / n, tot_edge / n, tot_perc / n, tot_last / n, last_outs, last_x


@torch.no_grad()
def evaluate_p1(model, loader, sobel, device, k_top=1,
                reveal_mode='tau', reveal_soft=0.05, gamma=0.9,
                perceptual_loss=None, w_perceptual=0.0, k_win=2):
    """P1 验证：(x_c, x_s) → x_gt。"""
    model.eval()
    tot_pix, tot_edge, tot_perc, tot_last, n = 0.0, 0.0, 0.0, 0.0, 0
    last_outs, last_xc, last_xs, last_gt = None, None, None, None
    for x_c, x_s, x_gt in loader:
        x_c, x_s, x_gt = x_c.to(device), x_s.to(device), x_gt.to(device)
        with autocast('cuda'):
            outs = model(x_c, x_s)
            ld = compute_losses(outs, x_gt, sobel, k_top=k_top,
                                reveal_mode=reveal_mode, reveal_soft=reveal_soft,
                                gamma=gamma,
                                perceptual_loss=perceptual_loss,
                                w_perceptual=w_perceptual,
                                k_win=k_win)
        bs = x_gt.size(0)
        tot_pix += ld['pix'].item() * bs
        tot_edge += ld['edge'].item() * bs
        tot_perc += ld['perceptual'].item() * bs
        tot_last += torch.nn.functional.l1_loss(
            outs[-1].float(), x_gt.float()).item() * bs
        n += bs
        last_outs, last_xc, last_xs, last_gt = outs, x_c, x_s, x_gt
    model.train()
    return (tot_pix / n, tot_edge / n, tot_perc / n, tot_last / n,
            last_outs, last_xc, last_xs, last_gt)


@torch.no_grad()
def evaluate_diffusion(model, loader, sobel, device, is_p1):
    """扩散验证：DDIM 采样后比较末步 x0 与目标。返回采样轨迹用于可视化。"""
    Fn = torch.nn.functional
    model.eval()
    tot_pix, tot_edge, tot_last, n = 0.0, 0.0, 0.0, 0
    last = None
    for batch in loader:
        if is_p1:
            x_c, x_s, x_gt = (t.to(device) for t in batch)
        else:
            x_c, x_s, x_gt = batch[0].to(device), None, batch[0].to(device)
        traj = model.sample(x_c, x_s)
        x0 = traj[-1].float()
        bs = x_gt.size(0)
        tot_pix += Fn.mse_loss(x0, x_gt.float()).item() * bs
        tot_edge += Fn.l1_loss(sobel(x0), sobel(x_gt.float())).item() * bs
        tot_last += Fn.l1_loss(x0, x_gt.float()).item() * bs
        n += bs
        last = (traj, x_c, x_s, x_gt)
    model.train()
    return tot_pix / n, tot_edge / n, tot_last / n, last


def run_diffusion(args, device, out_dir, is_p1, train_loader, val_loader):
    """循环去噪重建训练入口：x0-预测扩散，外层采样循环=循环深度。"""
    sobel = SobelEdge().to(device)
    model = RecurrentDiffusionRecon(
        T=args.T, T_inner=args.diff_inner, c_z=args.c_z,
        attn_dim=args.sp_attn_dim, heads=args.heads, c_canvas=args.channels,
        canvas_hw=16, c_style_mid=args.c_style_mid, c_style_deep=args.c_style_deep,
        c_content_aux=args.c_content_aux, c_content_main=args.c_content_main,
        dist_lambda=args.dist_lambda).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'[model] RecurrentDiffusionRecon T={args.T} T_inner={args.diff_inner} '
          f'c_z={args.c_z} attn_dim={args.sp_attn_dim} '
          f'params={n_total/1e6:.2f}M device={device}')
    perceptual = None
    if args.w_perceptual > 0:
        from losses import VGGPerceptualLoss
        perceptual = VGGPerceptualLoss().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    best_val = float('inf')
    T = args.T
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        agg = {'total': 0.0, 'pix': 0.0, 'edge': 0.0, 'perceptual': 0.0}
        n_seen = 0
        for batch in train_loader:
            if is_p1:
                x_c, x_s, x_gt = (t.to(device) for t in batch)
            else:
                x_c, x_s, x_gt = batch[0].to(device), None, batch[0].to(device)
            B = x_gt.size(0)
            t = torch.randint(1, T + 1, (B,), device=device)
            noise = torch.randn_like(x_gt)
            x_t = model.q_sample(x_gt, t, noise)
            cond = model.encode_cond(x_c, x_s)
            x0_prev = torch.zeros_like(x_gt)
            if torch.rand(()) < 0.5:                  # self-conditioning
                with torch.no_grad():
                    x0_prev = model.denoise_step(x_t, t, cond, x0_prev).clamp(-1, 1)
            x0_hat = model.denoise_step(x_t, t, cond, x0_prev)
            ld = diffusion_losses(x0_hat, x_gt, sobel,
                                  w_pix=args.diff_w_pix, w_edge=args.diff_w_edge,
                                  fg_weight=args.diff_fg_w,
                                  perceptual_loss=perceptual,
                                  w_perceptual=args.w_perceptual)
            optim.zero_grad(set_to_none=True)
            ld['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            for k in ('total', 'pix', 'edge', 'perceptual'):
                agg[k] += ld[k].item() * B
            n_seen += B
        for k in agg:
            agg[k] /= max(1, n_seen)

        val_pix, val_edge, val_last, last = evaluate_diffusion(
            model, val_loader, sobel, device, is_p1)
        dt = time.time() - t0
        print(f'[epoch {epoch:03d}] {dt:5.1f}s | train total={agg["total"]:.4f} '
              f'pix={agg["pix"]:.4f} edge={agg["edge"]:.4f} '
              f'perc={agg["perceptual"]:.4f} | '
              f'val mse={val_pix:.4f} edge={val_edge:.4f} last={val_last:.4f}')

        if epoch % args.vis_every == 0 or epoch == args.epochs:
            traj, vxc, vxs, vgt = last
            if is_p1:
                save_grid(traj, vgt, out_dir / f'vis_epoch{epoch:03d}.png',
                          extras=[vxc, vxs])
            else:
                save_grid(traj, vgt, out_dir / f'vis_epoch{epoch:03d}.png')
        if val_last < best_val:
            best_val = val_last
            torch.save({'model': model.state_dict(), 'args': vars(args),
                        'epoch': epoch}, out_dir / 'best.pt')
    print(f'[done] best val last-step L1 = {best_val:.4f}')


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_p1 = (args.mode == 'p1')

    # ---- 数据集 ----
    if is_p1:
        authors = args.authors if args.authors else args.author
        train_ds = StylePairDataset(args.root, args.struct_root, authors, 'train', args.size,
                                    n_refs=args.n_refs)
        val_ds   = StylePairDataset(args.root, args.struct_root, authors, 'val',   args.size,
                                    n_refs=args.n_refs, fixed_refs=True)
        if args.smoke:
            train_ds.items = train_ds.items[:32]
            val_ds.items   = val_ds.items[:8]
        if args.max_train_items > 0 and len(train_ds.items) > args.max_train_items:
            train_ds.items = train_ds.items[:args.max_train_items]
        print(f'[data/p1] authors={train_ds.authors} ({len(train_ds.authors)})  '
              f'train={len(train_ds)}  val={len(val_ds)}')
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_pair,
                                  drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate_pair)
    else:
        train_ds = MiniReconDataset(args.root, args.author, 'train', args.size)
        val_ds   = MiniReconDataset(args.root, args.author, 'val',   args.size)
        if args.smoke:
            train_ds.files = train_ds.files[:32]
            val_ds.files   = val_ds.files[:8]
        if args.max_train_items > 0 and len(train_ds.files) > args.max_train_items:
            train_ds.files = train_ds.files[:args.max_train_items]
        print(f'[data/p0] author={args.author}  train={len(train_ds)}  val={len(val_ds)}')
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate)

    # ---- 循环去噪重建分支：独立训练/评估，复用 trunk，不走 CTM/GAN 路径 ----
    if args.diffusion:
        run_diffusion(args, device, out_dir, is_p1, train_loader, val_loader)
        return

    # ---- 模型 ----
    is_hybrid = (args.model_type == 'hybrid')

    if is_hybrid:
        # CTMHybridRecon 同时支持 P0 和 P1
        model = CTMHybridRecon(
            T=args.T, M=args.M, c_z=args.c_z,
            sync_pairs=args.sp_sync_pairs, attn_dim=args.sp_attn_dim,
            heads=args.heads, c_canvas=args.channels, canvas_hw=16,
            c_style_mid=args.c_style_mid, c_style_deep=args.c_style_deep,
            c_content_aux=args.c_content_aux,
            c_content_main=args.c_content_main,
            r_init=args.r_init,
            dist_lambda=args.dist_lambda,
            style_amp=args.style_amp,
        ).to(device)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f'[model] CTMHybridRecon T={args.T} c_z={args.c_z} '
              f'sync_pairs={args.sp_sync_pairs} attn_dim={args.sp_attn_dim} '
              f'style_amp={args.style_amp} '
              f'trainable={n_trainable/1e6:.2f}M total={n_total/1e6:.2f}M  device={device}')
        print(f'[loss] T={args.T} reveal={args.reveal_mode} soft={args.reveal_soft} '
              f'gamma={args.gamma} k_win={args.k_win}')

        # 可选：加载对比预训练的 style backbone 权重
        if args.style_pretrained:
            ckpt = torch.load(args.style_pretrained, map_location='cpu')
            bb_sd = ckpt['backbone'] if 'backbone' in ckpt else ckpt
            msg = model.style_agg.enc.load_state_dict(bb_sd, strict=False)
            print(f'[pretrain] loaded style backbone from {args.style_pretrained} '
                  f'missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}')

        # 冻结 style backbone
        if args.freeze_style_epochs > 0:
            for p in model.style_agg.enc.parameters():
                p.requires_grad = False
            print(f'[freeze] style backbone frozen for first {args.freeze_style_epochs} epochs')

    elif is_p1:
        print('[warning] standard+p1: SpatialCTMRecon 已移除，使用 CTMHybridRecon')
        model = CTMHybridRecon(
            T=args.T, M=args.M, c_z=args.c_z,
            sync_pairs=args.sp_sync_pairs, attn_dim=args.sp_attn_dim,
            heads=args.heads, c_canvas=args.channels, canvas_hw=16,
            c_style_mid=args.c_style_mid, c_style_deep=args.c_style_deep,
            c_content_aux=args.c_content_aux,
            c_content_main=args.c_content_main,
            r_init=args.r_init,
            dist_lambda=args.dist_lambda,
            style_amp=args.style_amp,
        ).to(device)
        # 可选：加载对比预训练的 style backbone 权重
        if args.style_pretrained:
            ckpt = torch.load(args.style_pretrained, map_location='cpu')
            bb_sd = ckpt['backbone'] if 'backbone' in ckpt else ckpt
            msg = model.style_agg.enc.load_state_dict(bb_sd, strict=False)
            print(f'[pretrain] loaded style backbone from {args.style_pretrained} '
                  f'missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}')
        # 冻结 style backbone
        if args.freeze_style_epochs > 0:
            for p in model.style_agg.enc.parameters():
                p.requires_grad = False
            print(f'[freeze] style backbone frozen for first {args.freeze_style_epochs} epochs')
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f'[model] CTMHybridRecon T={args.T} c_z={args.c_z} '
              f'sync_pairs={args.sp_sync_pairs} attn_dim={args.sp_attn_dim} '
              f'style_amp={args.style_amp} '
              f'trainable={n_trainable/1e6:.2f}M total={n_total/1e6:.2f}M  device={device}')
        print(f'[loss] T={args.T} reveal={args.reveal_mode} soft={args.reveal_soft} '
              f'gamma={args.gamma} k_win={args.k_win} w_perceptual={args.w_perceptual} '
              f'(weighted deep supervise, VGG19 relu2_2+relu3_3 on last tick)')
    else:
        model = CTMRecon(c=args.channels, T=args.T, M=args.M, heads=args.heads,
                         single_branch=True, neurons=args.neurons,
                         sync_pairs=args.sync_pairs, attn_dim=args.attn_dim).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'[model] mode=p0 single_branch=True '
              f'neurons={args.neurons}  sync_pairs={args.sync_pairs}  '
              f'params={n_params/1e6:.2f}M  device={device}')

    sobel = SobelEdge().to(device)
    perceptual = None
    if is_p1 and args.w_perceptual > 0:
        from losses import VGGPerceptualLoss
        perceptual = VGGPerceptualLoss().to(device)
    # P1：G/D 双优化器；P0：单优化器
    optim_g = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    if is_p1:
        # conditional D：输入拼接 [x_c, image]，让 D 检查笔画是否对应印刷体字形
        netD = PatchDiscriminator(in_ch=2, base=args.d_base).to(device)
        optim_d = torch.optim.AdamW(netD.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
        n_d = sum(p.numel() for p in netD.parameters())
        print(f'[model] netD PatchDiscriminator(cGAN) in_ch=2 base={args.d_base} '
              f'params={n_d/1e6:.2f}M  w_adv={args.w_adv}')
    else:
        netD = None
        optim_d = None

    best_val = float('inf')
    scaler = GradScaler('cuda')
    has_style_backbone = (is_hybrid or is_p1)
    style_unfrozen = not (has_style_backbone and args.freeze_style_epochs > 0)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # 到达指定 epoch 后解冻 style backbone
        if (not style_unfrozen) and epoch > args.freeze_style_epochs:
            for p in model.style_agg.enc.parameters():
                p.requires_grad = True
            style_unfrozen = True
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'[unfreeze] epoch {epoch}: style backbone unfrozen, '
                  f'trainable={n_trainable/1e6:.2f}M')
        model.train()
        agg = {'total': 0.0, 'pix': 0.0, 'edge': 0.0, 'think': 0.0,
               'adv': 0.0, 'd': 0.0, 'perceptual': 0.0}
        n_seen = 0

        if is_p1:
            for x_c, x_s, x_gt in train_loader:
                x_c, x_s, x_gt = x_c.to(device), x_s.to(device), x_gt.to(device)

                # ---- D 步：固定 G 训练 D（conditional：D 看到 [x_c, image]） ----
                with torch.no_grad(), autocast('cuda'):
                    outs_d = model(x_c, x_s)
                    fake_d = outs_d[-1]
                with autocast('cuda'):
                    d_real = netD(torch.cat([x_c, x_gt], dim=1))
                    d_fake = netD(torch.cat([x_c, fake_d], dim=1))
                    loss_d = lsgan_d_loss(d_real, d_fake)
                optim_d.zero_grad(set_to_none=True)
                scaler.scale(loss_d).backward()
                scaler.step(optim_d)

                # ---- G 步：固定 D 训练 G ----
                # 对抗损失仅作用于最后 1 个 tick 的输出
                with autocast('cuda'):
                    outs = model(x_c, x_s)
                    fake = outs[-1]
                    d_fake_for_g = netD(torch.cat([x_c, fake], dim=1))
                    loss_adv = lsgan_g_loss(d_fake_for_g)
                    ld = compute_losses(outs, x_gt, sobel, k_top=args.k_top,
                                        reveal_mode=args.reveal_mode,
                                        reveal_soft=args.reveal_soft,
                                        gamma=args.gamma,
                                        perceptual_loss=perceptual,
                                        w_perceptual=args.w_perceptual,
                                        k_win=args.k_win)
                    loss_g = ld['total'] + args.w_adv * loss_adv
                optim_g.zero_grad(set_to_none=True)
                scaler.scale(loss_g).backward()
                scaler.unscale_(optim_g)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim_g)
                scaler.update()

                bs = x_gt.size(0)
                for k in ('total', 'pix', 'edge', 'think', 'perceptual'):
                    agg[k] += ld[k].item() * bs
                agg['adv'] += loss_adv.item() * bs
                agg['d'] += loss_d.item() * bs
                n_seen += bs
        else:
            for x, _ in train_loader:
                x = x.to(device)
                with autocast('cuda'):
                    outs = model(x)
                    ld = compute_losses(outs, x, sobel, k_top=args.k_top,
                                        reveal_mode=args.reveal_mode,
                                        reveal_soft=args.reveal_soft,
                                        gamma=args.gamma,
                                        perceptual_loss=perceptual,
                                        w_perceptual=args.w_perceptual,
                                        k_win=args.k_win)
                optim_g.zero_grad(set_to_none=True)
                scaler.scale(ld['total']).backward()
                scaler.unscale_(optim_g)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim_g)
                scaler.update()
                bs = x.size(0)
                for k in ('total', 'pix', 'edge', 'think', 'perceptual'):
                    agg[k] += ld[k].item() * bs
                n_seen += bs

        for k in agg:
            agg[k] /= max(1, n_seen)

        # ---- 验证 & 日志 ----
        if is_p1:
            val_pix, val_edge, val_perc, val_last, vo, vxc, vxs, vgt = evaluate_p1(
                model, val_loader, sobel, device, k_top=args.k_top,
                reveal_mode=args.reveal_mode, reveal_soft=args.reveal_soft,
                gamma=args.gamma,
                perceptual_loss=perceptual, w_perceptual=args.w_perceptual,
                k_win=args.k_win)
            ref_gt = vgt
        else:
            val_pix, val_edge, val_perc, val_last, vo, vx = evaluate_p0(
                model, val_loader, sobel, device, k_top=args.k_top,
                reveal_mode=args.reveal_mode, reveal_soft=args.reveal_soft,
                gamma=args.gamma,
                perceptual_loss=perceptual, w_perceptual=args.w_perceptual,
                k_win=args.k_win)
            ref_gt = vx

        dt = time.time() - t0
        if is_p1:
            print(f'[epoch {epoch:03d}] {dt:5.1f}s | train pix={agg["pix"]:.4f} '
                  f'edge={agg["edge"]:.4f} think={agg["think"]:.4f} '
                  f'perc={agg["perceptual"]:.4f} adv={agg["adv"]:.4f} d={agg["d"]:.4f} | '
                  f'val pix={val_pix:.4f} edge={val_edge:.4f} perc={val_perc:.4f} '
                  f'last={val_last:.4f}')
        else:
            print(f'[epoch {epoch:03d}] {dt:5.1f}s | train pix={agg["pix"]:.4f} '
                  f'edge={agg["edge"]:.4f} think={agg["think"]:.4f} '
                  f'perc={agg["perceptual"]:.4f} | '
                  f'val pix={val_pix:.4f} edge={val_edge:.4f} perc={val_perc:.4f} '
                  f'last={val_last:.4f}')

        with torch.no_grad():
            per_tick = [torch.nn.functional.l1_loss(o, ref_gt).item() for o in vo]
        print('           per-tick val L1: ' + ' -> '.join(f'{v:.4f}' for v in per_tick))

        if epoch % args.vis_every == 0 or epoch == args.epochs:
            if is_p1:
                save_grid(vo, vgt, out_dir / f'vis_epoch{epoch:03d}.png',
                          extras=[vxc, vxs])
            else:
                save_grid(vo, vx, out_dir / f'vis_epoch{epoch:03d}.png')

        if val_last < best_val:
            best_val = val_last
            torch.save({'model': model.state_dict(), 'args': vars(args), 'epoch': epoch},
                       out_dir / 'best.pt')

    print(f'[done] best val last-tick L1 = {best_val:.4f}')


if __name__ == '__main__':
    main()
