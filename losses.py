"""P0 损失函数：像素 L1 + Sobel 边缘 L1 + 思考一致性。

外部接口：
    compute_losses(outs, target, weights) -> dict[str,Tensor]
    其中 outs 是 model 返回的每个 tick 的输出列表（list of [B,1,H,W]）。
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label as _cc_label
import torchvision


class SobelEdge(nn.Module):
    """Sobel 边缘提取，返回 [B,1,H,W] 的梯度幅值。"""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        ky = kx.t().contiguous()
        kernel = torch.stack([kx, ky], dim=0).unsqueeze(1)   # [2,1,3,3]
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        g = F.conv2d(x, self.kernel, padding=1)              # [B,2,H,W]
        return torch.sqrt(g.pow(2).sum(dim=1, keepdim=True) + 1e-3)


class VGGPerceptualLoss(nn.Module):
    """基于 VGG-19 的感知损失，提取 relu2_2 和 relu3_3 层特征进行 L1 对比。

    输入图像预期范围为 [-1, 1]，内部先映射到 [0,1]再做 ImageNet 归一化。
    """

    def __init__(self, layer_indices=(8, 15), weights=(1.0, 1.0)):
        super().__init__()
        try:
            vgg = torchvision.models.vgg19(weights='DEFAULT').features
        except TypeError:                       # 兼容旧版 torchvision
            vgg = torchvision.models.vgg19(pretrained=True).features
        self.slices = nn.ModuleList()
        prev = 0
        for idx in layer_indices:
            self.slices.append(vgg[prev:idx + 1])
            prev = idx + 1
        for p in self.parameters():
            p.requires_grad = False
        self.weights = weights
        self.eval()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,H,W] in [-1,1] -> ImageNet normalized 3-channel
        x = (x + 1.0) / 2.0
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (x.repeat(1, 3, 1, 1) - mean) / std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self.normalize(pred)
        target = self.normalize(target)
        loss = 0.0
        for slice_net, w in zip(self.slices, self.weights):
            pred = slice_net(pred)
            with torch.no_grad():
                target = slice_net(target)
            loss = loss + w * F.l1_loss(pred, target)
        return loss


def pixel_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def edge_l1(pred: torch.Tensor, target: torch.Tensor, sobel: SobelEdge) -> torch.Tensor:
    return F.l1_loss(sobel(pred), sobel(target))


def think_consistency(outs: List[torch.Tensor]) -> torch.Tensor:
    """相邻 tick 输出的 L1 距离，抑制思考过程剧烈震荡。"""
    if len(outs) < 2:
        return outs[0].new_zeros(())
    loss = outs[0].new_zeros(())
    for t in range(1, len(outs)):
        loss = loss + F.l1_loss(outs[t], outs[t - 1])
    return loss / (len(outs) - 1)


@torch.no_grad()
def compute_tau_map(target: torch.Tensor, ink_threshold: float = 0.0) -> torch.Tensor:
    """逐像素揭示时间 τ ∈ [0, 1]，背景置 +inf。

    - 仅 ink 像素（target < ink_threshold）参与排序。
    - 连通分量（8-neighborhood）按面积降序排序，先大块后小细节。
    - 每个 CC 内按到质心距离升序排序，从中心向外扩张。
    - 线性面积分配：第 i 个 ink 像素 τ_i = (i+1) / N_ink，保证每 tick 揭示等量。
    """
    B, _, H, W = target.shape
    target_np = target.detach().float().cpu().numpy()
    tau = np.full((B, 1, H, W), np.inf, dtype=np.float32)
    structure = np.ones((3, 3), dtype=bool)

    for b in range(B):
        img = target_np[b, 0]
        ink = img < ink_threshold
        if not ink.any():
            continue
        labels, n_cc = _cc_label(ink, structure=structure)

        # 收集每个 CC 的面积、质心、像素坐标
        cc_list = []
        for cc_id in range(1, n_cc + 1):
            ys, xs = np.where(labels == cc_id)
            area = ys.size
            cy = float(ys.mean())
            cx = float(xs.mean())
            cc_list.append((area, cy, cx, ys, xs))
        cc_list.sort(key=lambda c: -c[0])  # 大 CC 优先

        rank_map = np.full((H, W), -1, dtype=np.int32)
        cur = 0
        for area, cy, cx, ys, xs in cc_list:
            d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
            order = np.argsort(d, kind='stable')
            rank_map[ys[order], xs[order]] = cur + np.arange(area, dtype=np.int32)
            cur += area

        if cur > 0:
            tau_b = np.where(
                rank_map >= 0,
                (rank_map.astype(np.float32) + 1.0) / float(cur),
                np.inf,
            )
            tau[b, 0] = tau_b

    return torch.from_numpy(tau).to(target.device, dtype=target.dtype)


def spatial_reveal_mask(
    t: int,
    T: int,
    H: int,
    W: int,
    mode: str = 'tau',
    soft: float = 0.05,
    tau_map: Optional[torch.Tensor] = None,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """构造空间渐进式揭示 mask。

    - mode='tau'：基于 compute_tau_map 的逐像素揭示时间，沿连通分量从内向外
      线性扩张；soft 单位为 fraction-of-progress（建议 0.03 ~ 0.08）。返回 [B,1,H,W]。
    - mode='center_out' / 'edge_in'：几何圆形扩张（ablation 用）；soft 单位为像素。
      返回 [1, 1, H, W]。
    所有模式值域均在 (0, 1) 之间。
    """
    progress = ((t + 1) / T) * (1.0 + 5.0 * soft) if T > 0 else 1.0
    if mode == 'tau':
        if tau_map is None:
            raise ValueError("tau_map is required when mode='tau'")
        return torch.sigmoid((progress - tau_map) / max(soft, 1e-3))

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij',
    )
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = float(torch.sqrt(torch.tensor(cy ** 2 + cx ** 2)).item())

    if mode == 'edge_in':
        inner = (1.0 - progress) * r_max
        mask = torch.sigmoid((dist - inner) / max(soft, 1e-3))
    else:  # center_out
        r_t = progress * r_max
        mask = torch.sigmoid((r_t - dist) / max(soft, 1e-3))
    return mask.view(1, 1, H, W)


def compute_losses(
    outs: List[torch.Tensor],
    target: torch.Tensor,
    sobel: SobelEdge,
    w_pix: float = 5.0,
    w_edge: float = 10.0,
    w_think: float = 0.0,
    deep_supervise_last: int = 0,            # 保留接口；新版 edge 已纳入逐 tick 循环
    ctm_loss: bool = True,                  # 保留接口；新版无效
    k_top: int = 1,                          # 保留接口；新版无效
    reveal_mode: str = 'tau',
    reveal_soft: float = 0.05,
    gamma: float = 0.9,
    bg_value: float = 1.0,
    perceptual_loss: Optional[nn.Module] = None,
    w_perceptual: float = 0.0,
    k_win: int = 2,
) -> Dict[str, torch.Tensor]:
    """加权深度监督 + Masked Target + τ-based Spatial Reveal：
    - reveal_mode='tau' 使用 compute_tau_map 的逐像素揭示时间（CC 顺序 +
      CC 内由心向外 + 线性面积增长）。
    - k_win > 0（默认 2）：滚动窗口监督——tick t 仅监督 (M_t - M_{t-k_win})
      最近 k_win 步新揭露的 ink 区域；前 k_win 个 tick 退化为累计揭露
      M_t 作为 warmup；最后 1 个 tick 始终解除 mask 用完整 target，保证
      outs[-1] 仍可作为终端全图监督和评估目标。
      此模式强迫 CTM 维护时间状态：旧 ink 必须"擦回白色"，本质是 CTM
      时间动力学压力测试。
    - k_win <= 0：回退到累计揭露（旧 (a) 行为）。每 tick target 为
      M_t * GT + (1 - M_t) * bg_value。
    - 衰减权重 w_t = γ^(T-1-t)，归一化后求和：后期 tick 权重大，但所有 16
      个 tick 都获得直接梯度信号。
    - edge L1 按 tick 在 target_t 上计算，与 pix L1 训练方向一致；使用
      与 pix 相同的 γ 衰减加权。
    - perceptual_loss 仅作用于最后 1 个 tick，与 pix L1 解除 mask 的
      范围保持一致。
    """
    T = len(outs)
    H, W = target.shape[-2], target.shape[-1]
    white = torch.full_like(target, bg_value)

    # 一次性算 τ map，供前部 tick 共享
    tau_map = compute_tau_map(target) if reveal_mode == 'tau' else None

    # 衰减权重：w_t = γ^(T-1-t)，归一化
    weights = torch.tensor(
        [gamma ** (T - 1 - t) for t in range(T)],
        device=target.device, dtype=torch.float32,
    )
    weights = weights / weights.sum()

    # 滚动窗口模式：预算 t=0..T-2 的累计 mask（最后 1 个 tick 走 full target 分支，无需 mask）
    use_window = k_win > 0
    mask_cum = None
    if use_window:
        mask_cum = [
            spatial_reveal_mask(
                t, T, H, W, mode=reveal_mode, soft=reveal_soft,
                tau_map=tau_map,
                device=target.device, dtype=target.dtype,
            )
            for t in range(T - 1)
        ]

    L_per_list = []
    E_per_list = []
    for t, o in enumerate(outs):
        # 仅最后 1 个 tick 解除 mask，直接使用完整 target
        if t == T - 1:
            mask_t = torch.ones_like(target)
            target_t = target
        elif use_window:
            if t < k_win:
                # warmup：前 k_win 个 tick 用累计揭露
                mask_t = mask_cum[t]
            else:
                # 滚动窗口：仅监督最近 k_win 步新揭露的 ink
                mask_t = (mask_cum[t] - mask_cum[t - k_win]).clamp_min(0)
            target_t = mask_t * target + (1.0 - mask_t) * white
        else:
            mask_t = spatial_reveal_mask(
                t, T, H, W, mode=reveal_mode, soft=reveal_soft,
                tau_map=tau_map,
                device=target.device, dtype=target.dtype,
            )
            target_t = mask_t * target + (1.0 - mask_t) * white
        diff = F.l1_loss(o, target_t, reduction='none')
        # 分别计算前景(被揭示部分)和背景(未揭示部分)的平均 Loss，消除面积不平衡
        loss_fg = (mask_t * diff).sum(dim=(1, 2, 3)) / mask_t.sum(dim=(1, 2, 3)).clamp_min(1e-6)
        loss_bg = ((1.0 - mask_t) * diff).sum(dim=(1, 2, 3)) / (1.0 - mask_t).sum(dim=(1, 2, 3)).clamp_min(1e-6)
        L_t = loss_fg + loss_bg
        L_per_list.append(L_t)
        # edge L1 同样基于 target_t，与 pix 方向一致；逐样本 mean 便于加权
        E_t = F.l1_loss(sobel(o), sobel(target_t), reduction='none').mean(dim=(1, 2, 3))
        E_per_list.append(E_t)

    L_per = torch.stack(L_per_list)                                  # [T, B]
    E_per = torch.stack(E_per_list)                                  # [T, B]
    # 加权深度监督：所有 tick 都拿到梯度，γ 控制后期 tick 主导程度
    w = weights.to(L_per.dtype).unsqueeze(1)                         # [T, 1]
    pix = (L_per * w).sum(dim=0).mean()
    edg = (E_per.to(w.dtype) * w).sum(dim=0).mean()

    think = think_consistency(outs)
    total = w_pix * pix + w_edge * edg + w_think * think

    # 感知损失：仅对最后一个完全 reveal 的 tick 计算
    perc = target.new_zeros(())
    if perceptual_loss is not None and w_perceptual > 0 and T > 0:
        perc = perceptual_loss(outs[-1], target)
        total = total + w_perceptual * perc

    return {'total': total, 'pix': pix, 'edge': edg, 'think': think, 'perceptual': perc}



def lsgan_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """LSGAN 判别器损失：MSE(D(real), 1) + MSE(D(fake), 0)。"""
    return 0.5 * (((d_real - 1.0) ** 2).mean() + (d_fake ** 2).mean())


def lsgan_g_loss(d_fake_for_g: torch.Tensor) -> torch.Tensor:
    """LSGAN 生成器损失：MSE(D(fake), 1)，让 G 骗过 D。"""
    return ((d_fake_for_g - 1.0) ** 2).mean()


def diffusion_losses(
    x0_hat: torch.Tensor,
    x_gt: torch.Tensor,
    sobel: SobelEdge,
    w_pix: float = 1.0,
    w_edge: float = 0.5,
    fg_weight: float = 2.0,
    ink_thr: float = 0.0,
    perceptual_loss: Optional[nn.Module] = None,
    w_perceptual: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """x0-prediction 扩散损失：前景加权 x0 MSE + Sobel 边缘 L1 (+ 可选感知)。

    x0_hat / x_gt 均为 [B,1,H,W]，范围 [-1,1]，背景 +1、字迹 -1。
    前景加权：ink 像素（x_gt < ink_thr）与背景分别按各自面积归一化平方误差，
    消除"背景占多数→细笔画梯度被淹没"导致的缺笔；fg_weight 进一步放大前景项。
    """
    se = (x0_hat - x_gt) ** 2                                       # [B,1,H,W]
    fg = (x_gt < ink_thr).to(se.dtype)
    fg_sum = fg.sum(dim=(1, 2, 3)).clamp_min(1.0)
    bg_sum = (1.0 - fg).sum(dim=(1, 2, 3)).clamp_min(1.0)
    loss_fg = (fg * se).sum(dim=(1, 2, 3)) / fg_sum
    loss_bg = ((1.0 - fg) * se).sum(dim=(1, 2, 3)) / bg_sum
    pix = (fg_weight * loss_fg + loss_bg).mean()
    edge = F.l1_loss(sobel(x0_hat), sobel(x_gt))
    total = w_pix * pix + w_edge * edge
    out = {'pix': pix, 'edge': edge}
    if perceptual_loss is not None and w_perceptual > 0:
        perc = perceptual_loss(x0_hat, x_gt)
        total = total + w_perceptual * perc
        out['perceptual'] = perc
    else:
        out['perceptual'] = pix.new_zeros(())
    out['total'] = total
    return out
