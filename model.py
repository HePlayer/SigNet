"""CTM-Hybrid 模型：CTM 循环深度架构 + 渐进式风格注入。

核心改进：
1. Q融合：每 tick 的 query = content_tokens + progress * tanh(q_ctm)，
   确保早期 tick 以内容为主，后期逐渐注入风格。
2. 风格放大：在 TwoStreamCrossAttn 门控后乘以 (1 + alpha * progress)，
   让风格 influence 随 tick 渐进增强。
3. PixelShuffleDecoder：用 PixelShuffle 替代 bilinear 上采样，
   消除网格伪影，提升笔画边界质量。
4. 加权深度监督 + Spatial Reveal：所有 16 个 tick 都获得梯度信号，
   中间 tick 用揭示掩码迫使模型逐步构建笔画。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 基础组件
# ============================================================================


class ConvBlock(nn.Module):
    def __init__(self, ci, co, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, k, s, p),
            nn.GroupNorm(8, co),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = ConvBlock(c, c)
        self.b2 = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.GroupNorm(8, c))

    def forward(self, x):
        return F.gelu(x + self.b2(self.b1(x)))


class ChannelLayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


def _fixed_sync_pairs(neurons: int, n_pairs: int, seed: int):
    """固定随机采样 neuron pair，用 register_buffer 保存，不参与学习。"""
    rows, cols = torch.triu_indices(neurons, neurons)
    total = rows.numel()
    n_pairs = min(n_pairs, total)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(total, generator=g)[:n_pairs]
    return torch.stack([rows[idx], cols[idx]], dim=1).long()


# ============================================================================
# 标准 CTM（P0 重建用，保持向后兼容）
# ============================================================================


class TokenCrossAttn(nn.Module):
    def __init__(self, dim=64, heads=4):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dk = dim // heads
        self.scale = self.dk ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, q, k_tokens, v_tokens):
        B, N, D = k_tokens.shape
        q = self.q_proj(q).view(B, self.heads, 1, self.dk)
        k = self.k_proj(k_tokens).view(B, N, self.heads, self.dk).transpose(1, 2)
        v = self.v_proj(v_tokens).view(B, N, self.heads, self.dk).transpose(1, 2)
        attn = torch.einsum('bhqd,bhnd->bhqn', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhqn,bhnd->bhqd', attn, v)
        out = out.transpose(1, 2).reshape(B, D)
        return self.o_proj(out)


class StandardCTMCore(nn.Module):
    def __init__(
        self,
        c=64,
        T=6,
        M=4,
        neurons=128,
        attn_dim=64,
        heads=4,
        sync_pairs=64,
        canvas_hw=16,
        pair_seed=1234,
    ):
        super().__init__()
        self.c, self.T, self.M = c, T, M
        self.neurons = neurons
        self.canvas_hw = canvas_hw

        self.z0 = nn.Parameter(torch.zeros(1, neurons))
        nn.init.normal_(self.z0, std=0.02)

        self.register_buffer(
            'pairs_action', _fixed_sync_pairs(neurons, sync_pairs, pair_seed)
        )
        self.register_buffer(
            'pairs_out', _fixed_sync_pairs(neurons, sync_pairs, pair_seed + 1)
        )

        self.sync_norm_action = nn.LayerNorm(sync_pairs)
        self.sync_norm_out = nn.LayerNorm(sync_pairs)
        self.to_query = nn.Linear(sync_pairs, attn_dim)
        self.attn = TokenCrossAttn(attn_dim, heads)

        hidden = max(neurons, attn_dim) * 2
        self.synapse = nn.Sequential(
            nn.Linear(neurons + attn_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, neurons),
        )
        self.a_norm = nn.LayerNorm(neurons)

        self.nlm = nn.Sequential(
            nn.Conv1d(neurons, neurons, kernel_size=M, groups=neurons),
            nn.GELU(),
            nn.Conv1d(neurons, neurons, kernel_size=1, groups=neurons),
        )
        self.z_norm = nn.LayerNorm(neurons)

        self.to_canvas = nn.Linear(sync_pairs, c * canvas_hw * canvas_hw)

    def _sync_repr(self, z_hist, pairs, norm):
        Z = torch.stack(z_hist, dim=-1)
        zi = Z[:, pairs[:, 0], :]
        zj = Z[:, pairs[:, 1], :]
        s = (zi * zj).mean(dim=-1)
        return norm(s)

    def _padded_pre_history(self, a_hist):
        A = torch.stack(a_hist[-self.M:], dim=-1)
        if A.size(-1) < self.M:
            pad = A.new_zeros(A.size(0), A.size(1), self.M - A.size(-1))
            A = torch.cat([pad, A], dim=-1)
        return A

    def forward(self, k_tokens, v_tokens, decoder):
        B = k_tokens.size(0)
        z = self.z0.expand(B, -1)
        z_hist = [z]
        a_hist = []
        outs = []

        for _ in range(self.T):
            s_action = self._sync_repr(z_hist, self.pairs_action, self.sync_norm_action)
            q = self.to_query(s_action)
            o = self.attn(q, k_tokens, v_tokens)

            a = self.synapse(torch.cat([z, o], dim=-1))
            a = self.a_norm(a)
            a_hist.append(a)
            A = self._padded_pre_history(a_hist)

            z = self.nlm(A).squeeze(-1)
            z = self.z_norm(z)
            z_hist.append(z)

            s_out = self._sync_repr(z_hist, self.pairs_out, self.sync_norm_out)
            canvas = self.to_canvas(s_out).view(
                B, self.c, self.canvas_hw, self.canvas_hw
            ) / math.sqrt(self.c)
            outs.append(decoder(canvas))

        return outs


class CTMRecon(nn.Module):
    def __init__(
        self,
        c=64,
        T=6,
        M=4,
        heads=4,
        single_branch=True,
        neurons=128,
        sync_pairs=64,
        attn_dim=64,
        canvas_hw=16,
    ):
        super().__init__()
        self.single = single_branch
        self.enc_c = Encoder(c)
        self.enc_s = self.enc_c if single_branch else Encoder(c)
        self.k_proj = nn.Conv2d(c, attn_dim, 1)
        self.v_proj = nn.Conv2d(c, attn_dim, 1)
        self.pos = nn.Parameter(torch.zeros(1, attn_dim, canvas_hw, canvas_hw))
        self.core = StandardCTMCore(
            c=c,
            T=T,
            M=M,
            neurons=neurons,
            attn_dim=attn_dim,
            heads=heads,
            sync_pairs=sync_pairs,
            canvas_hw=canvas_hw,
        )
        self.decoder = Decoder(c)

    def _tokens(self, feat, proj):
        x = proj(feat)
        pos = self.pos
        if pos.shape[-2:] != x.shape[-2:]:
            pos = F.interpolate(pos, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + pos
        return x.flatten(2).transpose(1, 2)

    def forward(self, x_c, x_s=None):
        K_feat = self.enc_c(x_c)
        V_feat = K_feat if (self.single or x_s is None) else self.enc_s(x_s)
        K = self._tokens(K_feat, self.k_proj)
        V = self._tokens(V_feat, self.v_proj)
        return self.core(K, V, self.decoder)


# ============================================================================
# 内容编码器（用于 CTM-Hybrid）
# ============================================================================


class LearnableContentEncoder(nn.Module):
    """印刷体内容编码器：浅层可学习卷积，专门为汉字结构提取设计。

    输入 [B,1,64,64] -> 输出两层特征：
        k_aux  [B, c_aux,  32, 32]
        k_main [B, c_main, 16, 16]
    """

    def __init__(self, in_ch: int = 1, c_aux: int = 64, c_main: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 7, 1, 3),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(32, c_aux, 4, 2, 1),
            nn.GroupNorm(8, c_aux),
            nn.GELU(),
            ResBlock(c_aux),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c_aux, c_main, 4, 2, 1),
            nn.GroupNorm(8, c_main),
            nn.GELU(),
            ResBlock(c_main),
        )

    def forward(self, x):
        x = self.stem(x)
        f_aux = self.down1(x)
        f_main = self.down2(f_aux)
        return f_aux, f_main


# ============================================================================
# 风格编码器（用于 CTM-Hybrid）
# ============================================================================


class StyleBackbone(nn.Module):
    """更深的风格编码器：每个下采样 stage 后增加 ResBlock 提升表征能力。"""

    def __init__(self, c_mid: int = 64, c_deep: int = 96):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBlock(1, c_mid // 2, k=4, s=2, p=1),
            ResBlock(c_mid // 2),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(c_mid // 2, c_mid, k=4, s=2, p=1),
            ResBlock(c_mid),
            ResBlock(c_mid),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(c_mid, c_deep, k=4, s=2, p=1),
            ResBlock(c_deep),
            ResBlock(c_deep),
        )

    def forward(self, x):
        f1 = self.stage1(x)
        f_mid = self.stage2(f1)
        f_deep = self.stage3(f_mid)
        return f_mid, f_deep


class StyleAggregator(nn.Module):
    """对 N 张参考字做多尺度自适应池化与跨参考聚合，输出 V tokens。

    每张参考字独立编码（不跨 N 取 mean），让 attention 自行选择最相关的风格特征。
    """

    def __init__(self, attn_dim: int = 96, c_mid: int = 64, c_deep: int = 96,
                 pool_mid: int = 4, pool_deep: int = 2):
        super().__init__()
        self.enc = StyleBackbone(c_mid=c_mid, c_deep=c_deep)
        self.pool_mid = pool_mid
        self.pool_deep = pool_deep
        self.proj_mid = nn.Conv2d(c_mid, attn_dim, 1)
        self.proj_deep = nn.Conv2d(c_deep, attn_dim, 1)
        self.scale_emb = nn.Parameter(torch.zeros(2, attn_dim))
        nn.init.normal_(self.scale_emb, std=0.02)

    def forward(self, x_s):
        B, N, C, H, W = x_s.shape
        x = x_s.reshape(B * N, C, H, W)
        f_mid, f_deep = self.enc(x)
        t_mid = self.proj_mid(F.adaptive_avg_pool2d(f_mid, self.pool_mid))
        t_deep = self.proj_deep(F.adaptive_avg_pool2d(f_deep, self.pool_deep))
        D = t_mid.size(1)
        t_mid = t_mid.flatten(2).transpose(1, 2)
        t_deep = t_deep.flatten(2).transpose(1, 2)
        t_mid = t_mid.reshape(B, N * t_mid.size(1), D) + self.scale_emb[0]
        t_deep = t_deep.reshape(B, N * t_deep.size(1), D) + self.scale_emb[1]
        return torch.cat([t_mid, t_deep], dim=1)


# ============================================================================
# 注意力模块（含风格放大）
# ============================================================================


class SpatialCrossAttn(nn.Module):
    """空间化 cross-attention：Q 是特征图 [B,d,H,W]，KV 是 tokens [B,N,d]"""

    def __init__(self, dim: int, heads: int, dist_lambda: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dk = dim // heads
        self.scale = self.dk ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.dist_lambda = dist_lambda
        self.register_buffer('_dist_matrix', torch.zeros(0))

    def _get_dist_matrix(self, H: int, W: int, device: torch.device):
        if self._dist_matrix.numel() > 0:
            dm = self._dist_matrix
            if dm.shape[-2:] == (H * W, H * W) and dm.device == device:
                return dm
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        coords = torch.stack([yy, xx], dim=-1).reshape(H * W, 2)
        dist = torch.cdist(coords, coords, p=2).unsqueeze(0).unsqueeze(0)
        self._dist_matrix = dist
        return dist

    def forward(self, q_map, kv_tokens):
        B, d, H, W = q_map.shape
        Hd, dk = self.heads, self.dk
        q = q_map.permute(0, 2, 3, 1).reshape(B, H * W, d)
        q = self.q_proj(q).reshape(B, H * W, Hd, dk).transpose(1, 2)
        N = kv_tokens.size(1)
        k = self.k_proj(kv_tokens).reshape(B, N, Hd, dk).transpose(1, 2)
        v = self.v_proj(kv_tokens).reshape(B, N, Hd, dk).transpose(1, 2)

        if self.dist_lambda > 0 and N == H * W:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            dist = self._get_dist_matrix(H, W, q_map.device)
            scores = scores - self.dist_lambda * dist.to(scores.dtype)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        out = out.transpose(1, 2).reshape(B, H * W, d)
        out = self.o_proj(out)
        return out.reshape(B, H, W, d).permute(0, 3, 1, 2).contiguous()


class TwoStreamCrossAttn(nn.Module):
    """K 来自 x_c、V 来自 x_s 的双流交叉注意力，按通道门控融合。

    新增风格放大：门控融合后乘以 (1 + style_amp * progress)，
    progress 由外部传入（t / (T-1)），让风格 influence 随 tick 渐进增强。
    """

    def __init__(self, dim: int, heads: int, dist_lambda: float = 0.05,
                 gate_init_bias: float = -2.0, style_amp: float = 2.0):
        super().__init__()
        self.attn_c = SpatialCrossAttn(dim, heads, dist_lambda=dist_lambda)
        self.attn_s = SpatialCrossAttn(dim, heads)
        self.gate = nn.Conv2d(dim * 2, dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init_bias)
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.style_amp = style_amp

    def forward(self, q_map, content_map, style_tokens, progress=1.0):
        B, d, H, W = content_map.shape
        content_tokens = content_map.permute(0, 2, 3, 1).reshape(B, H * W, d)
        o_c = self.attn_c(q_map, content_tokens)
        o_s = self.attn_s(q_map, style_tokens)
        g = torch.sigmoid(self.gate(torch.cat([o_c, o_s], dim=1)))
        style_factor = 1.0 + self.style_amp * progress
        return self.norm(o_c * (1 - g) + o_s * g * style_factor)


class SpatialSelfAttn(nn.Module):
    """空间自注意：每 tick 做全局自注意，弥补 3x3 邻域传播不足。

    out_proj 与 FFN 末层零初始化，使模块插入瞬间等价 identity。
    """

    def __init__(self, c: int, H: int, W: int,
                 num_heads: int = 4, ffn_ratio: int = 2):
        super().__init__()
        assert c % num_heads == 0
        self.c, self.H, self.W = c, H, W

        self.pos = nn.Parameter(torch.zeros(1, H * W, c))
        nn.init.normal_(self.pos, std=0.02)

        self.norm1 = nn.LayerNorm(c)
        self.mha = nn.MultiheadAttention(c, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(
            nn.Linear(c, c * ffn_ratio),
            nn.GELU(),
            nn.Linear(c * ffn_ratio, c),
        )

        nn.init.zeros_(self.mha.out_proj.weight)
        nn.init.zeros_(self.mha.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2) + self.pos
        h = self.norm1(tokens)
        attn_out, _ = self.mha(h, h, h, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).view(B, C, H, W)


# ============================================================================
# CTM-Hybrid Core（含 Q 融合）
# ============================================================================


class CTMHybridCore(nn.Module):
    """空间化 CTM-Hybrid：每个 (h, w) 位置独立维护神经元同步。

    关键改进：
    1. Q融合：q_map = content_map + progress * tanh(q_ctm)，
       早期 tick 近似 identity（仅内容），后期逐渐注入 CTM 思考结果。
    2. 时序衰减按一阶递推（Eq.16/17）高效更新：
        α_{ij}^{t+1} = exp(-r_{ij}) · α_{ij}^t + z_i^{t+1} · z_j^{t+1}
        β_{ij}^{t+1} = exp(-r_{ij}) · β_{ij}^t + 1
        S_{ij}^t     = α_{ij}^t / sqrt(β_{ij}^t)
    3. 内容锚定：每 tick NLM 更新后注入 content_proj(content_map)，
       防止 z 漂离印刷体结构。
    """

    def __init__(self, c_z: int = 128, H: int = 16, W: int = 16,
                 T: int = 16, M: int = 4, sync_pairs: int = 64,
                 attn_dim: int = 128, c_canvas: int = 64,
                 pair_seed: int = 1234, r_init: float = -3.0):
        super().__init__()
        self.c_z, self.H, self.W = c_z, H, W
        self.T, self.M, self.P = T, M, sync_pairs

        self.z0 = nn.Parameter(torch.zeros(1, c_z, 1, 1))
        nn.init.normal_(self.z0, std=0.02)

        self.register_buffer(
            'pairs_action', _fixed_sync_pairs(c_z, sync_pairs, pair_seed))
        self.register_buffer(
            'pairs_out', _fixed_sync_pairs(c_z, sync_pairs, pair_seed + 1))

        self.r_raw_action = nn.Parameter(torch.full((sync_pairs,), float(r_init)))
        self.r_raw_out = nn.Parameter(torch.full((sync_pairs,), float(r_init)))

        self.sync_norm_action = ChannelLayerNorm2d(sync_pairs)
        self.sync_norm_out = ChannelLayerNorm2d(sync_pairs)

        # Q 融合：从同步表示投影出 q_ctm
        self.to_query = nn.Conv2d(sync_pairs, attn_dim, 1)

        # sync_o -> canvas：先 1x1 投影，再用 3x3 + ResBlock 空间 refine
        self.to_canvas = nn.Sequential(
            nn.Conv2d(sync_pairs, c_canvas, 1),
            ConvBlock(c_canvas, c_canvas),
            ResBlock(c_canvas),
        )

        # synapse：DWSep 3x3 引入空间耦合后 pointwise 投影
        hidden = max(c_z + attn_dim, c_z * 2)
        self.synapse = nn.Sequential(
            nn.Conv2d(c_z + attn_dim, c_z + attn_dim, 3, 1, 1,
                      groups=c_z + attn_dim),
            nn.GELU(),
            nn.Conv2d(c_z + attn_dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, c_z, 1),
        )
        self.a_norm = ChannelLayerNorm2d(c_z)

        self.spatial_attn = SpatialSelfAttn(c_z, H, W)

        self.nlm = nn.Sequential(
            nn.Conv1d(c_z, c_z, kernel_size=M, groups=c_z),
            nn.GELU(),
            nn.Conv1d(c_z, c_z, kernel_size=1, groups=c_z),
        )
        self.z_norm = ChannelLayerNorm2d(c_z)

        self.content_proj = nn.Conv2d(attn_dim, c_z, 1)
        nn.init.zeros_(self.content_proj.weight)
        nn.init.zeros_(self.content_proj.bias)

    def _sync_step(self, alpha, beta, z, pairs, r_raw):
        r = F.softplus(r_raw)
        decay = torch.exp(-r)
        zi = z[:, pairs[:, 0]]
        zj = z[:, pairs[:, 1]]
        alpha = decay[None, :, None, None] * alpha + zi * zj
        beta = decay * beta + 1.0
        sync = alpha / torch.sqrt(beta)[None, :, None, None].clamp_min(1e-6)
        return alpha, beta, sync

    def forward(self, content_map, style_tokens, attn_module, z_init=None):
        B = content_map.size(0)
        H, W, D, P = self.H, self.W, self.c_z, self.P
        if z_init is not None:
            z = (z_init + self.z0).contiguous()
        else:
            z = self.z0.expand(B, -1, H, W).contiguous()

        a_hist = z.new_zeros(B, D, H * W, self.M)
        alpha_a = z.new_zeros(B, P, H, W)
        beta_a = z.new_zeros(P)
        alpha_o = z.new_zeros(B, P, H, W)
        beta_o = z.new_zeros(P)

        canvases = []
        T = self.T
        for t in range(T):
            progress = t / max(T - 1, 1)

            alpha_a, beta_a, sync_a = self._sync_step(
                alpha_a, beta_a, z, self.pairs_action, self.r_raw_action)
            sync_a = self.sync_norm_action(sync_a)
            q_ctm = self.to_query(sync_a)

            # ========== Q融合 ==========
            # content_map (B, attn_dim, H, W) 提供结构锚定
            # q_ctm 提供 CTM 内部同步的动态 query
            # 早期: q ≈ content_map（重建为主）
            # 后期: q ≈ content_map + tanh(q_ctm)（风格置入）
            q_map = content_map + progress * torch.tanh(q_ctm)

            o = attn_module(q_map, content_map, style_tokens, progress=progress)

            a = self.synapse(torch.cat([z, o], dim=1))
            a = self.a_norm(a)
            a = self.spatial_attn(a)

            a_flat = a.view(B, D, H * W)
            a_hist = torch.cat([a_hist[..., 1:], a_flat.unsqueeze(-1)], dim=-1)

            A = a_hist.permute(0, 2, 1, 3).reshape(B * H * W, D, self.M)
            z_new = self.nlm(A).squeeze(-1)
            z = z_new.view(B, H * W, D).transpose(1, 2).view(B, D, H, W)
            z = self.z_norm(z)
            z = z + self.content_proj(content_map)

            alpha_o, beta_o, sync_o = self._sync_step(
                alpha_o, beta_o, z, self.pairs_out, self.r_raw_out)
            sync_o = self.sync_norm_out(sync_o)
            canvases.append(self.to_canvas(sync_o))

        return canvases


# ============================================================================
# PixelShuffle 解码器
# ============================================================================


class PixelShuffleDecoder(nn.Module):
    """从 16x16 canvas + 内容多尺度 skip 解码到 64x64。

    上采样用 PixelShuffle（depth-to-space），相比 bilinear 不引入
    固定平滑先验，让网络自行学习上采样核，消除网格伪影。
    """

    def __init__(self, c_canvas: int = 64, c_skip_main: int = 128,
                 c_skip_aux: int = 64, c_mid: int = 64):
        super().__init__()
        # 16x16: canvas + 内容主层 skip 融合
        self.fuse16 = nn.Sequential(
            nn.Conv2d(c_canvas + c_skip_main, c_mid, 1),
            nn.GroupNorm(8, c_mid), nn.GELU(),
            ResBlock(c_mid),
        )
        # 16 -> 32: PixelShuffle（通道×4，空间×2）
        self.up1 = nn.Sequential(
            nn.Conv2d(c_mid, c_mid * 4, 3, 1, 1),
            nn.GroupNorm(8, c_mid * 4),
            nn.GELU(),
            nn.PixelShuffle(2),
            ConvBlock(c_mid, c_mid),
            ResBlock(c_mid),
        )
        # 32x32: 与 aux skip 融合
        self.fuse32 = nn.Sequential(
            nn.Conv2d(c_mid + c_skip_aux, c_mid // 2, 1),
            nn.GroupNorm(8, c_mid // 2), nn.GELU(),
            ConvBlock(c_mid // 2, c_mid // 2),
        )
        # 32 -> 64: PixelShuffle
        self.up2 = nn.Sequential(
            nn.Conv2d(c_mid // 2, (c_mid // 2) * 4, 3, 1, 1),
            nn.GroupNorm(8, (c_mid // 2) * 4),
            nn.GELU(),
            nn.PixelShuffle(2),
            ConvBlock(c_mid // 2, c_mid // 2),
            ConvBlock(c_mid // 2, c_mid // 4),
        )
        self.to_img = nn.Sequential(
            nn.Conv2d(c_mid // 4, 1, 7, 1, 3),
            nn.Tanh(),
        )

    def forward(self, canvas, k_skip_main, k_skip_aux):
        x = self.fuse16(torch.cat([canvas, k_skip_main], dim=1))
        x = self.up1(x)
        x = self.fuse32(torch.cat([x, k_skip_aux], dim=1))
        x = self.up2(x)
        return self.to_img(x)


# ============================================================================
# CTM-Hybrid Recon（顶层模型）
# ============================================================================


class CTMHybridRecon(nn.Module):
    """CTM-Hybrid 顶层模型：K=印刷体，V=多参考风格聚合，Q=CTM 同步+内容融合。

    支持 P0（自重建）和 P1（风格迁移）两种模式：
    - P0：x_s = x_c（单输入自重建）
    - P1：(x_c, x_s) -> x_gt（印刷体+风格参考 -> 目标手写体）

    关键设计：
    - Q融合：每 tick q = content_map + progress * tanh(q_ctm)
    - 风格放大：门控后乘以 (1 + style_amp * progress)
    - PixelShuffle 解码器
    - 加权深度监督 + Spatial Reveal（由 train.py 的 loss 实现）
    """

    def __init__(self, T: int = 16, M: int = 4, c_z: int = 192,
                 sync_pairs: int = 80, attn_dim: int = 128, heads: int = 4,
                 c_canvas: int = 64, canvas_hw: int = 16,
                 c_style_mid: int = 96, c_style_deep: int = 160,
                 c_content_aux: int = 64, c_content_main: int = 128,
                 r_init: float = -3.0, dist_lambda: float = 0.05,
                 style_amp: float = 2.0):
        super().__init__()
        self.canvas_hw = canvas_hw
        self.T = T

        # 内容编码器
        self.content_enc = LearnableContentEncoder(
            in_ch=1, c_aux=c_content_aux, c_main=c_content_main)
        self.k_proj = nn.Conv2d(c_content_main, attn_dim, 1)
        self.k_pos = nn.Parameter(torch.zeros(1, attn_dim, canvas_hw, canvas_hw))
        nn.init.normal_(self.k_pos, std=0.02)

        # 从内容特征投影出空间化的初始 z
        self.z_init_proj = nn.Conv2d(c_content_main, c_z, 1)

        # 风格聚合器
        self.style_agg = StyleAggregator(
            attn_dim=attn_dim, c_mid=c_style_mid, c_deep=c_style_deep,
            pool_mid=4, pool_deep=2)

        # 双流交叉注意力（含风格放大）
        self.attn = TwoStreamCrossAttn(
            dim=attn_dim, heads=heads, dist_lambda=dist_lambda,
            style_amp=style_amp)

        # CTM-Hybrid 核心
        self.core = CTMHybridCore(
            c_z=c_z, H=canvas_hw, W=canvas_hw, T=T, M=M,
            sync_pairs=sync_pairs, attn_dim=attn_dim, c_canvas=c_canvas,
            r_init=r_init)

        # PixelShuffle 解码器
        self.decoder = PixelShuffleDecoder(
            c_canvas=c_canvas, c_skip_main=c_content_main,
            c_skip_aux=c_content_aux, c_mid=64)

    def forward(self, x_c, x_s=None):
        # x_c: [B,1,64,64]; x_s: [B,N,1,64,64] or None (P0 mode)
        k_aux, k_main = self.content_enc(x_c)
        content_map = self.k_proj(k_main) + self.k_pos

        if x_s is not None:
            style_tokens = self.style_agg(x_s)
        else:
            # P0 模式：以自身为风格参考
            B = x_c.size(0)
            style_tokens = content_map.flatten(2).transpose(1, 2)

        z_init = self.z_init_proj(k_main)
        canvases = self.core(content_map, style_tokens, self.attn, z_init=z_init)
        outs = [self.decoder(c, k_main, k_aux) for c in canvases]
        return outs


# ============================================================================
# PatchGAN 判别器（P1 对抗训练用）
# ============================================================================


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN，输入 [B, in_ch, H, W]，输出 [B, 1, H/2^4, W/2^4]。"""

    def __init__(self, in_ch: int = 2, base: int = 32):
        super().__init__()
        def _blk(ci, co, s=2):
            return nn.Sequential(
                nn.Conv2d(ci, co, 4, s, 1),
                nn.GroupNorm(8, co),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            _blk(base, base * 2),
            _blk(base * 2, base * 4),
            _blk(base * 4, base * 8),
            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)



# ============================================================================
# 循环去噪重建（Recurrent Denoising Refinement）：在像素空间做 x0-预测扩散，
# 外层采样循环 = 循环深度（迭代上一结果）；trunk 复用 cross-attn / synapse /
# SpatialSelfAttn / PixelShuffleDecoder，去掉 sync 的 EMA 低通与硬内容锚定。
# ============================================================================


def cosine_abar(T: int, s: float = 0.008) -> torch.Tensor:
    """cosine 噪声调度的 alpha_cumprod，长度 T+1，abar[0]=1（干净）。"""
    t = torch.linspace(0, 1, T + 1)
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    return abar.clamp(1e-5, 1.0)


class TimeEmb(nn.Module):
    """正弦时间步嵌入 + MLP，输出 [B, out]。"""

    def __init__(self, dim: int = 128, out: int = 192):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, out), nn.GELU(), nn.Linear(out, out))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None].float() * freqs[None]
        emb = torch.cat([a.sin(), a.cos()], dim=-1)
        return self.mlp(emb)


class NoisyEncoder(nn.Module):
    """带噪图 + 自条件 → 16x16 隐特征：[B,1,64,64]x2 -> [B,c_z,16,16]。"""

    def __init__(self, c_z: int = 192):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, 7, 1, 3), nn.GroupNorm(8, 32), nn.GELU())
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 96, 4, 2, 1), nn.GroupNorm(8, 96), nn.GELU(),
            ResBlock(96))
        self.down2 = nn.Sequential(
            nn.Conv2d(96, c_z, 4, 2, 1), nn.GroupNorm(8, c_z), nn.GELU(),
            ResBlock(c_z))

    def forward(self, x_t, x0_prev):
        x = torch.cat([x_t, x0_prev], dim=1)
        return self.down2(self.down1(self.stem(x)))


class DenoiseCore(nn.Module):
    """单步去噪 trunk：复用 cross-attn + synapse + SpatialSelfAttn，
    无 EMA、无硬内容锚；content 仅作 cross-attn 条件。跑 T_inner 步残差更新。"""

    def __init__(self, c_z: int = 192, H: int = 16, W: int = 16,
                 T_inner: int = 1, attn_dim: int = 128, c_canvas: int = 64):
        super().__init__()
        self.T_inner = T_inner
        self.to_query = nn.Conv2d(c_z, attn_dim, 1)
        hidden = max(c_z + attn_dim, c_z * 2)
        self.synapse = nn.Sequential(
            nn.Conv2d(c_z + attn_dim, c_z + attn_dim, 3, 1, 1,
                      groups=c_z + attn_dim),
            nn.GELU(),
            nn.Conv2d(c_z + attn_dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, c_z, 1),
        )
        self.a_norm = ChannelLayerNorm2d(c_z)
        self.spatial_attn = SpatialSelfAttn(c_z, H, W)
        self.z_norm = ChannelLayerNorm2d(c_z)
        self.to_canvas = nn.Sequential(
            nn.Conv2d(c_z, c_canvas, 1),
            ConvBlock(c_canvas, c_canvas),
            ResBlock(c_canvas),
        )

    def forward(self, content_map, style_tokens, attn_module, z_init, t_emb):
        z = z_init + t_emb[:, :, None, None]
        for _ in range(self.T_inner):
            q = content_map + torch.tanh(self.to_query(z))
            o = attn_module(q, content_map, style_tokens, progress=1.0)
            h = self.synapse(torch.cat([z, o], dim=1))
            h = self.a_norm(h)
            h = self.spatial_attn(h)
            z = self.z_norm(z + h)
        return self.to_canvas(z)


class RecurrentDiffusionRecon(nn.Module):
    """顶层：印刷体+风格参考为条件，循环去噪生成手写体。

    支持 P0（x_s=None，自身为风格参考）与 P1（多参考）。
    """

    def __init__(self, T: int = 16, T_inner: int = 1, c_z: int = 192,
                 attn_dim: int = 128, heads: int = 4, c_canvas: int = 64,
                 canvas_hw: int = 16, c_style_mid: int = 96,
                 c_style_deep: int = 160, c_content_aux: int = 64,
                 c_content_main: int = 128, dist_lambda: float = 0.05):
        super().__init__()
        self.T = T
        self.content_enc = LearnableContentEncoder(
            in_ch=1, c_aux=c_content_aux, c_main=c_content_main)
        self.k_proj = nn.Conv2d(c_content_main, attn_dim, 1)
        self.k_pos = nn.Parameter(torch.zeros(1, attn_dim, canvas_hw, canvas_hw))
        nn.init.normal_(self.k_pos, std=0.02)
        self.style_agg = StyleAggregator(
            attn_dim=attn_dim, c_mid=c_style_mid, c_deep=c_style_deep,
            pool_mid=4, pool_deep=2)
        self.attn = TwoStreamCrossAttn(
            dim=attn_dim, heads=heads, dist_lambda=dist_lambda,
            gate_init_bias=0.0, style_amp=0.0)
        self.core = DenoiseCore(
            c_z=c_z, H=canvas_hw, W=canvas_hw, T_inner=T_inner,
            attn_dim=attn_dim, c_canvas=c_canvas)
        self.decoder = PixelShuffleDecoder(
            c_canvas=c_canvas, c_skip_main=c_content_main,
            c_skip_aux=c_content_aux, c_mid=64)
        self.noisy_enc = NoisyEncoder(c_z)
        self.time_emb = TimeEmb(out=c_z)
        self.register_buffer('abar', cosine_abar(T))

    def encode_cond(self, x_c, x_s=None):
        k_aux, k_main = self.content_enc(x_c)
        content_map = self.k_proj(k_main) + self.k_pos
        if x_s is not None:
            style_tokens = self.style_agg(x_s)
        else:
            style_tokens = content_map.flatten(2).transpose(1, 2)
        return content_map, style_tokens, k_main, k_aux

    def q_sample(self, x0, t, noise):
        a = self.abar[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1.0 - a).sqrt() * noise

    def denoise_step(self, x_t, t, cond, x0_prev):
        content_map, style_tokens, k_main, k_aux = cond
        z_init = self.noisy_enc(x_t, x0_prev)
        canvas = self.core(content_map, style_tokens, self.attn,
                           z_init, self.time_emb(t))
        return self.decoder(canvas, k_main, k_aux)

    @torch.no_grad()
    def sample(self, x_c, x_s=None):
        cond = self.encode_cond(x_c, x_s)
        B = x_c.size(0)
        x = torch.randn_like(x_c)
        x0_prev = torch.zeros_like(x_c)
        traj = []
        for ti in range(self.T, 0, -1):
            t = torch.full((B,), ti, device=x_c.device, dtype=torch.long)
            x0 = self.denoise_step(x, t, cond, x0_prev).clamp(-1, 1)
            a_t = self.abar[ti]
            a_prev = self.abar[ti - 1]
            eps = (x - a_t.sqrt() * x0) / (1.0 - a_t).clamp_min(1e-5).sqrt()
            x = a_prev.sqrt() * x0 + (1.0 - a_prev).clamp_min(0).sqrt() * eps
            x0_prev = x0
            traj.append(x0)
        return traj