"""数据集模块：支持 P0 自重建和 P1 风格迁移。

P0 MiniReconDataset：
- 单作者，x→x 自重建，验证框架能否收敛。

P1 StylePairDataset：
- 三元组 (x_c, x_s, x_gt)：
    x_c  = 印刷体（StructDataset/char_{UNICODE}.jpg），提供 K（结构内容）
    x_s  = 同一作者随机 N 张其它字的手写体，shape=[N,1,S,S]，提供 V（风格参考）
    x_gt = 目标字符的手写体（监督信号）

约定：
- 输入图片为灰度 JPG，背景白、字迹黑。
- 等比例缩放后居中 pad 到 size×size，pad 值使用背景白 (255)。
- 归一化到 [-1, 1]：背景 +1，字迹 ~ -1。
- 训练/验证按 unicode 划分，固定 seed 保证可复现。
"""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def _load_info(info_path: Path) -> List[str]:
    """读取 info.txt，返回文件名列表（跳过表头）。"""
    files = []
    with info_path.open('r', encoding='utf-8') as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 5:
                files.append(parts[4])
    return files


def _resize_pad(img: Image.Image, size: int) -> np.ndarray:
    """等比例缩放最长边到 size，居中 pad 到 size×size。返回 [-1,1] float32。"""
    w, h = img.size
    s = size / max(w, h)
    new_w, new_h = max(1, int(round(w * s))), max(1, int(round(h * s)))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new('L', (size, size), 255)
    canvas.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    arr = np.asarray(canvas, dtype=np.float32) / 127.5 - 1.0
    return arr


class MiniReconDataset(Dataset):
    """单作者自重建数据集。

    Args:
        root: 数据根目录，例如 'data/StyleDataset'。
        author: 作者目录名，例如 '1126-f'。
        split: 'train' 或 'val'。
        size: 输出图像边长。
        val_ratio: 验证集比例。
        seed: 划分随机种子。
    """

    def __init__(
        self,
        root: str = 'data/StyleDataset',
        author: str = '1126-f',
        split: str = 'train',
        size: int = 64,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        assert split in ('train', 'val')
        self.dir = Path(root) / 'Train' / author
        self.size = size

        files = _load_info(self.dir / 'info.txt')
        # 仅保留实际存在的文件，避免脏数据
        files = [fn for fn in files if (self.dir / fn).is_file()]
        files.sort()

        rng = random.Random(seed)
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_ratio))
        self.files = files[n_val:] if split == 'train' else files[:n_val]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        fn = self.files[idx]
        img = Image.open(self.dir / fn).convert('L')
        arr = _resize_pad(img, self.size)            # [size,size]
        return arr[None], fn                          # [1,size,size]


def collate(batch):
    import torch
    imgs = np.stack([b[0] for b in batch], axis=0)    # [B,1,S,S]
    names = [b[1] for b in batch]
    return torch.from_numpy(imgs).float(), names


# ---------------------------------------------------------------------------
# P1：三元组数据集
# ---------------------------------------------------------------------------

def _load_info_with_unicode(info_path: Path):
    """读取 info.txt，返回 [(unicode_str, filename), ...] 列表（跳过表头）。"""
    items = []
    with info_path.open('r', encoding='utf-8') as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 5:
                items.append((parts[0].upper(), parts[4]))  # unicode 大写
    return items


class StylePairDataset(Dataset):
    """P1 风格迁移数据集：(x_c, x_s, x_gt) 三元组，支持单/多作者。

    Args:
        style_root:  StyleDataset 根目录，如 'data/StyleDataset'
        struct_root: StructDataset 根目录，如 'data/StructDataset'
        author:      作者：单个名（'1126-f'）、逗号分隔串（'1001-f,1002-f'）
                     或名字列表。多作者时风格参考 x_s 仅取同作者，印刷体 x_c 共享。
        split:       'train' 或 'val'
        size:        输出图像边长（默认 64）
        val_ratio:   验证集比例（按作者各自划分后合并）
        seed:        划分随机种子
    """

    def __init__(
        self,
        style_root: str = 'data/StyleDataset',
        struct_root: str = 'data/StructDataset',
        author='1126-f',
        split: str = 'train',
        size: int = 64,
        val_ratio: float = 0.1,
        seed: int = 42,
        n_refs: int = 4,
        fixed_refs: bool = False,
    ):
        assert split in ('train', 'val')
        if isinstance(author, str):
            authors = [a.strip() for a in author.split(',') if a.strip()]
        else:
            authors = [str(a).strip() for a in author]
        self.authors = authors
        self.style_root = Path(style_root) / 'Train'
        self.struct_dir = Path(struct_root)
        self.size = size
        self.n_refs = max(1, int(n_refs))
        self.fixed_refs = fixed_refs

        # 跨作者聚合：每个作者按 unicode 独立划分 train/val 后合并
        # item = (author, unicode, filename)
        items = []
        for ai, a in enumerate(authors):
            style_dir = self.style_root / a
            all_items = _load_info_with_unicode(style_dir / 'info.txt')
            valid_a = []
            for uni, fn in all_items:
                hw_path = style_dir / fn
                pr_path = self.struct_dir / f'char_{uni}.jpg'
                if hw_path.is_file() and pr_path.is_file():
                    valid_a.append((a, uni, fn))
            valid_a.sort()
            rng = random.Random(seed + ai)
            rng.shuffle(valid_a)
            n_val = max(1, int(len(valid_a) * val_ratio))
            sel = valid_a[n_val:] if split == 'train' else valid_a[:n_val]
            items.extend(sel)
        self.items = items

        # 验证集（或 fixed_refs）使用确定性参考索引，避免 val 波动
        self._fixed_seed = seed + 1
        self._fixed_table = None
        self._fixed_table_len = -1

    def __len__(self) -> int:
        return len(self.items)

    def _author_candidates(self, idx: int):
        """与 idx 同作者、且 != idx 的索引列表（基于当前 items，自适应截断）。"""
        a = self.items[idx][0]
        return [j for j, it in enumerate(self.items) if it[0] == a and j != idx]

    def _ensure_fixed_table(self):
        """懒构建固定参考表；items 长度变化时（smoke 截断）自动重建。
        参考仅取同作者，保证风格一致。"""
        n = len(self.items)
        if self._fixed_table is not None and self._fixed_table_len == n:
            return
        rng2 = random.Random(self._fixed_seed)
        # 预构建每个作者的索引池
        author_pool = {}
        for j, it in enumerate(self.items):
            author_pool.setdefault(it[0], []).append(j)
        table = []
        for i in range(n):
            pool = [j for j in author_pool[self.items[i][0]] if j != i]
            if not pool:
                pool = [i]
            rng2.shuffle(pool)
            reps = (self.n_refs // len(pool)) + 1
            table.append((pool * reps)[:self.n_refs])
        self._fixed_table = table
        self._fixed_table_len = n

    def _sample_ref_indices(self, idx: int):
        if self.fixed_refs:
            self._ensure_fixed_table()
            return self._fixed_table[idx]
        candidates = self._author_candidates(idx)
        if not candidates:
            return [idx] * self.n_refs
        if len(candidates) >= self.n_refs:
            return random.sample(candidates, self.n_refs)
        return [candidates[random.randint(0, len(candidates) - 1)]
                for _ in range(self.n_refs)]

    def __getitem__(self, idx: int):
        author, uni, fn = self.items[idx]

        # x_gt：目标字符的手写体
        x_gt = _resize_pad(
            Image.open(self.style_root / author / fn).convert('L'), self.size)

        # x_c：对应印刷体（K 来源，作者无关）
        x_c = _resize_pad(
            Image.open(self.struct_dir / f'char_{uni}.jpg').convert('L'), self.size
        )

        # x_s：同一作者 N 张其它字的手写体（V 来源），shape=[N,1,S,S]
        refs = self._sample_ref_indices(idx)
        x_s_list = []
        for j in refs:
            a2, _, s_fn = self.items[j]
            arr = _resize_pad(
                Image.open(self.style_root / a2 / s_fn).convert('L'), self.size)
            x_s_list.append(arr[None])
        x_s = np.stack(x_s_list, axis=0)          # [N,1,S,S]

        return x_c[None], x_s, x_gt[None]


def collate_pair(batch):
    """StylePairDataset 的 collate 函数，返回 (x_c, x_s, x_gt) Tensor。

    形状：x_c [B,1,S,S]，x_s [B,N,1,S,S]，x_gt [B,1,S,S]。
    """
    import torch
    x_c  = np.stack([b[0] for b in batch], axis=0)
    x_s  = np.stack([b[1] for b in batch], axis=0)
    x_gt = np.stack([b[2] for b in batch], axis=0)
    return (
        torch.from_numpy(x_c).float(),
        torch.from_numpy(x_s).float(),
        torch.from_numpy(x_gt).float(),
    )
