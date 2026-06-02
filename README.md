# Signet — 汉字手写风格迁移（循环去噪重建）

基于**循环去噪重建（Recurrent Denoising Refinement）**的中文手写风格迁移项目。
输入印刷体汉字（内容）与同一书写者的若干参考手写字（风格），在像素空间执行
**x0-预测扩散**，把噪声沿反向过程**逐步迭代精修**为目标书写者的手写体图像（64×64，灰度）。

> 当前主线为扩散精修（`RecurrentDiffusionRecon`）。早期 CTM 架构（`CTMHybridRecon`）
> 仍保留在代码中，见文末「附录：历史架构」。

---

## 任务定义

| 记号 | 来源 | 形状 / 含义 |
|------|------|------|
| $x_c$ | 印刷体 | $1\times64\times64$，内容 / 笔画骨架（cross-attn 的 Key） |
| $x_s$ | $N$ 张同作者参考手写体 | $N\times1\times64\times64$，书写风格（cross-attn 的 Value） |
| $x_0=x_{gt}$ | 目标字手写体 | $1\times64\times64$，监督信号，像素范围 $[-1,1]$ |
| $\hat x_0$ | 模型输出 | 对 $x_0$ 的预测（x0-prediction） |

**训练数据**：`data/StyleDataset`（手写体，按作者分目录）、`data/StructDataset`（印刷体 `char_{UNICODE}.jpg`）
**损失函数**：$x_0$ 的 MSE + Sobel 边缘 L1（+ 可选 VGG 感知）

---

## 核心思想：把「循环深度」诠释为扩散反向步

每一步去噪都以上一步的估计 $\hat x_0^{(prev)}$ 为**自条件**，迭代得到更优解。

**前向加噪（cosine 调度）**，令 $f(t)=\cos^2\!\big(\tfrac{t/T+s}{1+s}\cdot\tfrac{\pi}{2}\big)$：

$$\bar\alpha_t=\frac{f(t)}{f(0)},\qquad t=0,1,\dots,T$$

$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I)$$

**反向预测**：模型直接预测干净图 $x_0$（而非噪声 $\epsilon$），对小数据集与少步采样更稳定：

$$\hat x_0=\mathcal D_\theta\big(x_t,\,t,\,x_0^{(prev)};\,x_c,\,x_s\big)$$

---

## 架构数据流图（张量形状 + KQV 来源）

记号：$B$=batch，$N$=`n_refs`（同作者风格参考数，默认 8），$d$=`attn_dim`=128，
$C_z$=`c_z`=192；空间分辨率标在形状中。下图为**单个去噪步**的完整数据流。

```
输入
  x_c      [B,1,64,64]      印刷体（内容，KV-content 来源）
  x_s      [B,N,1,64,64]    同作者参考手写（风格，KV-style 来源）
  x_t      [B,1,64,64]      第 t 步带噪图
  x0_prev  [B,1,64,64]      上一步 x0 估计（自条件）
  t        [B]              时间步

━━ 条件编码 encode_cond（每次采样只算一次）━━━━━━━━━━━━━━━━━━━━━━━━━━━
 x_c ─► ContentEnc ─┬─► k_aux  [B,64,32,32] ───────────────────► decoder skip
                    └─► k_main [B,128,16,16] ─► conv1×1 +k_pos
                                                = M_content [B,128,16,16]
                                                ├──► (KV-content 来源)
                                                └──────────────► decoder skip
 x_s ─► StyleAgg ─► T_style [B, N·20, 128]   (每参考: 4×4 mid + 2×2 deep = 20)
                                                └──► (KV-style 来源)

━━ 带噪编码 + 时间嵌入 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [x_t ; x0_prev] [B,2,64,64] ─► NoisyEnc(conv7→down→down) ─► z_init [B,192,16,16]
 t [B] ─► TimeEmb(sinusoid→MLP) ─► τ [B,192] ──(+)──►  z [B,192,16,16]

━━ 去噪核 DenoiseCore（内部重复 T_inner 次，默认 1）━━━━━━━━━━━━━━━━━━━━
 ┌──────────────────────────────────────────────────────────────────────┐
 │ Q 构造:  q = M_content + tanh(conv1×1(z))            [B,128,16,16]     │
 │                                                                        │
 │ TwoStreamCrossAttn  (q 展平为 256 个空间位置作 Query):                 │
 │   ┌ 内容流 attn_c:  Q = q              [B,256,128]                     │
 │   │                 K,V = M_content    [B,256,128]   ◄── 来自 x_c      │
 │   │                 → o_c              [B,128,16,16]                   │
 │   └ 风格流 attn_s:  Q = q              [B,256,128]                     │
 │                     K,V = T_style      [B,N·20,128]  ◄── 来自 x_s      │
 │                     → o_s              [B,128,16,16]                   │
 │   门控融合: g=σ(conv1×1([o_c;o_s]));  o=GN(o_c·(1-g)+o_s·g)            │
 │                                        → o  [B,128,16,16]              │
 │                                                                        │
 │ Synapse([z;o]) [B,320,16,16] ─dwconv3→conv1→conv1─► h [B,192,16,16]   │
 │ h = SpatialSelfAttn(LN(h))                          [B,192,16,16]     │
 │ z = LN(z + h)                                       [B,192,16,16]     │
 └──────────────────────────────────────────────────────────────────────┘
 末步: canvas c = to_canvas(z) = ResBlock(ConvBlock(conv1×1(z)))  [B,64,16,16]

━━ 解码 PixelShuffleDecoder（含内容 skip）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 c [B,64,16,16] ─cat─ k_main[B,128,16,16] ─► fuse16        ─► [B,64,16,16]
                                            ─► PixelShuffle×2 ─► [B,64,32,32]
 [B,64,32,32]   ─cat─ k_aux [B,64,32,32]  ─► fuse32         ─► [B,32,32,32]
                                            ─► PixelShuffle×2 ─► [B,16,64,64]
                                            ─► conv7+Tanh    ─► x0_hat [B,1,64,64]

━━ DDIM 采样循环（t = T…1，T=16；自条件回灌）━━━━━━━━━━━━━━━━━━━━━━━━━
 x_T ~ N(0,I) [B,1,64,64];  x0_prev = 0
 for t = T..1:
   x0_hat = DenoiseStep(x_t, t, cond, x0_prev)           [B,1,64,64]
   eps    = (x_t − √ᾱ_t · x0_hat) / √(1−ᾱ_t)
   x_{t-1}= √ᾱ_{t-1} · x0_hat + √(1−ᾱ_{t-1}) · eps
   x0_prev ← x0_hat
 输出末步 x0_hat [B,1,64,64]
```

**KQV 来源速查**：

| 注意力 | Query 来源 | Key/Value 来源 | Q 形状 | K/V 形状 |
|--------|-----------|----------------|--------|----------|
| 内容流 `attn_c` | 去噪隐状态 $z$ → $q$（256 空间位置） | $M_{content}$（**印刷体 $x_c$**） | $[B,256,128]$ | $[B,256,128]$ |
| 风格流 `attn_s` | 同上 $q$ | $T_{style}$（**手写参考 $x_s$**） | $[B,256,128]$ | $[B,N{\cdot}20,128]$ |
| 空间自注意力 `SpatialSelfAttn` | Synapse 输出（自身） | 同图自身（256 位置） | $[B,256,192]$ | $[B,256,192]$ |

> 要点：**内容 $x_c$ 只进 K/V-content + 解码 skip**，**风格 $x_s$ 只进 K/V-style**，
> 二者都不直接作 Query；Query 始终来自去噪隐状态 $z$（即"思考查询"），
> 由门控 $g$ 决定每个空间位置上"听内容还是听风格"。

---

## 架构（层用函数表示）

约定记号：$\mathrm{conv}_k$ 为 $k\times k$ 卷积，$\mathrm{dwconv}_3$ 为 depthwise $3\times3$ 卷积，
$\mathrm{GN}$ 为 GroupNorm，$\mathrm{LN}$ 为逐位置 ChannelLayerNorm，$\mathrm{PS}{\times}2$ 为
PixelShuffle 上采样，$\sigma$ 为 sigmoid，$[\,\cdot\,;\,\cdot\,]$ 为通道拼接。

### 1. 条件编码 `encode_cond`

$$(k_{aux},k_{main})=\mathrm{ContentEnc}(x_c),\quad k_{aux}\in\mathbb R^{C_a\times32\times32},\ k_{main}\in\mathbb R^{C_m\times16\times16}$$

$$M_{content}=\mathrm{conv}_1(k_{main})+P_{pos}\in\mathbb R^{d\times16\times16}$$

$$T_{style}=\mathrm{StyleAgg}(x_s)\in\mathbb R^{B\times L\times d}\quad(\text{P0 模式 }x_s=\varnothing\text{ 时取 }M_{content}\text{ 展平})$$

### 2. 带噪编码 + 时间嵌入

$$z_{init}=\mathrm{NoisyEnc}(x_t,x_0^{(prev)})=\mathrm{down}_2\!\big(\mathrm{down}_1(\mathrm{stem}([x_t;x_0^{(prev)}]))\big)\in\mathbb R^{C_z\times16\times16}$$

其中 $\mathrm{stem}=\mathrm{GELU}\!\circ\!\mathrm{GN}\!\circ\!\mathrm{conv}_7$，$\mathrm{down}_i=\mathrm{ResBlock}\!\circ\!\mathrm{GELU}\!\circ\!\mathrm{GN}\!\circ\!\mathrm{conv}_{4,s2}$。时间嵌入：

$$\tau=\mathrm{MLP}\big(\mathrm{sinusoid}(t)\big)\in\mathbb R^{C_z}$$

### 3. 去噪核 `DenoiseCore`（残差精修 $T_{inner}$ 步）

初值 $z\leftarrow z_{init}+\tau$，每步：

$$q=M_{content}+\tanh(\mathrm{conv}_1(z))$$

$$o=\mathrm{CrossAttn}(q,\,M_{content},\,T_{style})$$

$$h=\mathrm{SpatialSelfAttn}\big(\mathrm{LN}(\mathrm{Synapse}([z;o]))\big),\qquad z=\mathrm{LN}(z+h)$$

末步输出 canvas：$\;c=\mathrm{ResBlock}(\mathrm{ConvBlock}(\mathrm{conv}_1(z)))\in\mathbb R^{C\times16\times16}$。

其中 Synapse 为深度可分离前馈：

$$\mathrm{Synapse}(u)=\mathrm{conv}_1\Big(\mathrm{GELU}\big(\mathrm{conv}_1(\mathrm{GELU}(\mathrm{dwconv}_3(u)))\big)\Big)$$

### 4. 双流交叉注意力 `TwoStreamCrossAttn`

内容流与风格流分别注意，再按通道门控融合（去掉了硬内容残差注入）：

$$o_c=\mathrm{Attn}(q,M_{content}),\quad o_s=\mathrm{Attn}(q,T_{style})$$

$$g=\sigma\big(\mathrm{conv}_1([o_c;o_s])\big),\qquad \mathrm{CrossAttn}=\mathrm{GN}\big(o_c\odot(1-g)+o_s\odot g\big)$$

### 5. 解码 `PixelShuffleDecoder`

$$\hat x_0=\tanh\Big(\mathrm{conv}_7\big(\mathrm{PS}{\times}2\big(\mathrm{fuse}_{32}\big([\,\mathrm{PS}{\times}2(\mathrm{fuse}_{16}([c;k_{main}]))\,;\,k_{aux}\,]\big)\big)\big)\Big)\in\mathbb R^{1\times64\times64}$$

### 6. 采样 `sample`（DDIM，$T$ 步）

初始化 $x_T\sim\mathcal N(0,I)$、$x_0^{(prev)}=0$；对 $t=T,\dots,1$：

$$\hat x_0=\mathcal D_\theta(x_t,t,x_0^{(prev)}),\qquad \hat\epsilon=\frac{x_t-\sqrt{\bar\alpha_t}\,\hat x_0}{\sqrt{1-\bar\alpha_t}}$$

$$x_{t-1}=\sqrt{\bar\alpha_{t-1}}\,\hat x_0+\sqrt{1-\bar\alpha_{t-1}}\,\hat\epsilon,\qquad x_0^{(prev)}\leftarrow\hat x_0$$

---

## 训练目标

每个样本随机时间步 $t\sim\mathcal U\{1,\dots,T\}$，以 50% 概率执行一次自条件前传得到 $x_0^{(prev)}$（其余为 0）：

$$\mathcal L=w_{pix}\,\big\lVert\hat x_0-x_0\big\rVert_2^2+w_{edge}\,\big\lVert\mathrm{Sobel}(\hat x_0)-\mathrm{Sobel}(x_0)\big\rVert_1\;(+\,w_{perc}\,\mathcal L_{VGG})$$

---

## 快速开始

```bash
# 循环去噪重建训练（P1 风格迁移，默认 6 作者 / 全部字）
python train.py --diffusion --mode p1 --epochs 100 --batch 16 --vis_every 5 --out_dir runs/diff_p1

# 指定单作者 P1
python train.py --diffusion --mode p1 --authors 1126-f --epochs 100 --batch 16

# 冒烟测试（1 epoch，验证环境与采样链路）
python train.py --diffusion --mode p1 --smoke

# P0 自重建（同一字作内容与目标，框架验证）
python train.py --diffusion --mode p0 --author 1126-f --epochs 40 --batch 16
```

主要超参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--diffusion` | off | 启用循环去噪重建分支（`RecurrentDiffusionRecon`） |
| `--authors` | 6 作者 | P1 多作者列表（逗号分隔）；空串则回退 `--author` |
| `--T` | 16 | 扩散步数 = 外层采样循环深度 |
| `--diff_inner` | 1 | `DenoiseCore` 每步内部残差精修次数 $T_{inner}$ |
| `--c_z` | 192 | 隐状态 $z$ 通道数 |
| `--sp_attn_dim` | 128 | cross-attn 维度 $d$ |
| `--n_refs` | 8 | 每样本同作者风格参考张数 $N$ |
| `--diff_w_pix` | 1.0 | $x_0$ 的 MSE 权重 $w_{pix}$ |
| `--diff_w_edge` | 0.5 | Sobel 边缘 L1 权重 $w_{edge}$ |
| `--w_perceptual` | 1.0 | VGG 感知损失权重 $w_{perc}$（0 关闭） |

---

## 文件结构

```
Signet/
├── model.py        # 模型组件：RecurrentDiffusionRecon（当前主线）/ CTMHybridRecon（历史）
├── train.py        # 训练入口：--diffusion 扩散分支 / 旧 CTM P0/P1 分支
├── datareader.py   # 数据集：MiniReconDataset / StylePairDataset（支持多作者）
├── losses.py       # 损失：diffusion_losses（x0 MSE + 边缘 + 感知）/ compute_losses
└── data/
    ├── StyleDataset/   # 手写体，Train/{author}/ 下按作者分目录
    └── StructDataset/  # 印刷体，char_{UNICODE}.jpg
```

---

## 附录：历史架构（CTM）

早期采用 **Continuous Thought Machine (CTM)** 的循环深度思想（`CTMHybridRecon`）：以 CTM
内部同步状态生成 cross-attention 的 Query，逐 tick 注入风格、窗口化深度监督。该路径仍可用
（去掉 `--diffusion` 即走旧分支），但因 L1 平均导致的笔画模糊（regression-to-mean）与内部
EMA 低通效应，已被像素空间扩散精修取代。

---

## 参考文献

- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020
- Song et al., *Denoising Diffusion Implicit Models* (DDIM), ICLR 2021
- Sakana AI, *Continuous Thought Machines*, arXiv:2505.05522v4 (2025)
