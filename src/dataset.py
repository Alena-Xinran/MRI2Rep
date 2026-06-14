# src/dataset.py

import os
import json
import torch
import numpy as np
import random
from torch.utils.data import Dataset, Sampler
from .config import Config, PAD
# [REMOVED] from .utils import stable_split  <-- 不再需要动态计算哈希


class Augmentor3D:
    """
    3D MRI augmentation for tensors of shape (3, H, W, D) = [ART, PV, MASK].
    - Intensity transforms (scale/shift/noise/gamma) applied only to ART/PV channels.
    - Spatial flip applied only along the depth (z) axis, which is label-safe:
      position labels (L_LAT, R_ANT, etc.) are in-plane anatomical regions that
      span the full superior-inferior extent, so z-axis flip does not change them.
    - Random scale-crop: crop H×W by a random factor then resize back.
      Label-safe because cropping preserves relative anatomy (L/R not swapped).
    - Smooth elastic deformation in H×W plane (not Z) via bilinear grid_sample.
      Displacement amplitude is kept small (~6px) so L/R boundaries are not crossed.
    """

    @staticmethod
    def _random_scale_crop(x: torch.Tensor, scale_range=(0.88, 1.0)) -> torch.Tensor:
        """
        Randomly crop H and W by a scale factor, then resize back to original size.
        This simulates mild zoom variation while preserving anatomical L/R topology.
        The MASK channel is re-binarized after interpolation.
        """
        C, H, W, D = x.shape
        scale = random.uniform(*scale_range)
        new_h = max(1, int(H * scale))
        new_w = max(1, int(W * scale))

        # Random crop position
        y0 = random.randint(0, H - new_h)
        x0 = random.randint(0, W - new_w)
        cropped = x[:, y0:y0 + new_h, x0:x0 + new_w, :]   # (C, new_h, new_w, D)

        # Resize back: F.interpolate expects (N, C, D, H, W)
        vol = cropped.unsqueeze(0).permute(0, 1, 4, 2, 3)  # (1, C, D, new_h, new_w)
        resized = torch.nn.functional.interpolate(
            vol.float(), size=(D, H, W), mode='trilinear', align_corners=True
        )  # (1, C, D, H, W)
        out = resized.squeeze(0).permute(0, 2, 3, 1)        # (C, H, W, D)

        # Re-binarize mask channel (interpolation makes it soft)
        out[2] = (out[2] > 0.5).float()
        return out

    @staticmethod
    def _elastic_deform(x: torch.Tensor, alpha: float = 6.0) -> torch.Tensor:
        """
        Apply smooth elastic deformation in the H×W plane only (not Z).
        Each depth slice gets the same displacement field so relative depth
        anatomy is fully preserved.  Alpha is the maximum displacement in pixels.

        Implementation: generate a coarse random offset grid, bicubic-upsample to
        full spatial size, then apply via grid_sample.  Both ART/PV (continuous)
        and MASK (binary) are warped; MASK is re-binarized afterwards.
        """
        C, H, W, D = x.shape

        # Random coarse displacement (normalized to [-1, 1] grid space)
        coarse = 10
        # alpha normalised: alpha pixels / (H or W / 2) converts to grid units
        scale_h = alpha / (H / 2.0)
        scale_w = alpha / (W / 2.0)
        dy = (torch.randn(1, 1, coarse, coarse) * scale_h)
        dx = (torch.randn(1, 1, coarse, coarse) * scale_w)

        # Upsample to full spatial size with smooth bicubic interpolation
        dy = torch.nn.functional.interpolate(
            dy, size=(H, W), mode='bicubic', align_corners=True).squeeze()  # (H, W)
        dx = torch.nn.functional.interpolate(
            dx, size=(H, W), mode='bicubic', align_corners=True).squeeze()  # (H, W)

        # Build sampling grid: base identity + displacement
        # grid_sample convention: grid[..., 0] = x (W), grid[..., 1] = y (H)
        base_x = torch.linspace(-1, 1, W).unsqueeze(0).expand(H, -1)   # (H, W)
        base_y = torch.linspace(-1, 1, H).unsqueeze(1).expand(-1, W)   # (H, W)
        grid = torch.stack([base_x + dx, base_y + dy], dim=-1)          # (H, W, 2)
        grid = grid.unsqueeze(0).expand(D, -1, -1, -1)                  # (D, H, W, 2)

        # Apply: treat D as batch dimension
        x_slices = x.permute(3, 0, 1, 2)   # (D, C, H, W)
        warped = torch.nn.functional.grid_sample(
            x_slices.float(), grid,
            mode='bilinear', padding_mode='border', align_corners=True
        )   # (D, C, H, W)
        out = warped.permute(1, 2, 3, 0)    # (C, H, W, D)

        # Re-binarize mask channel
        out[2] = (out[2] > 0.5).float()
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: float tensor, shape (3, H, W, D), values in [0, 1]."""
        x = x.clone()

        # ── 1. Spatial: flip along the depth axis (dim=3, superior-inferior) ──
        if random.random() < 0.5:
            x = torch.flip(x, dims=[3])

        # ── 2. Random scale-crop in H×W (label-safe zoom augmentation) ──────────
        if random.random() < 0.4:
            x = self._random_scale_crop(x, scale_range=(0.88, 1.0))

        # ── 3. Smooth elastic deformation in H×W (label-safe local warp) ────────
        if random.random() < 0.3:
            x = self._elastic_deform(x, alpha=6.0)

        # ── 4. Intensity: ART/PV channels (0, 1) only ──────────────────────────

        # Global multiplicative scale (simulate global brightness variation)
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            x[:2] = x[:2] * scale

        # Global additive shift (simulate DC offset)
        if random.random() < 0.3:
            shift = random.uniform(-0.05, 0.05)
            x[:2] = x[:2] + shift

        # Per-channel scale (ART vs PV contrast difference varies between scanners)
        for c in range(2):
            if random.random() < 0.3:
                x[c] = x[c] * random.uniform(0.92, 1.08)

        # Additive Gaussian noise (simulate scanner noise)
        if random.random() < 0.3:
            std = random.uniform(0.01, 0.03)
            x[:2] = x[:2] + torch.randn_like(x[:2]) * std

        # Gamma correction (simulate non-linear intensity response)
        if random.random() < 0.2:
            gamma = random.uniform(0.85, 1.15)
            x[:2] = torch.clamp(x[:2], 1e-6, 1.0).pow(gamma)

        # Clamp to valid range
        x[:2] = torch.clamp(x[:2], 0.0, 1.0)
        # Mask channel stays binary – no intensity transform applied

        return x


class MRISeqDataset(Dataset):
    def __init__(self, cfg: Config, split: str):
        self.cfg = cfg
        self.split = split  # 'train', 'val', or 'test'
        
        # 1. 确定 Cache 目录
        self.cache_dir = os.path.join(cfg.cache_root, cfg.cache_tag)
        manifest_path = os.path.join(self.cache_dir, "manifest.json")
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                f"Please run 'python preprocess_data.py --tag {cfg.cache_tag}' first."
            )
            
        # 2. 加载 Manifest 并筛选
        with open(manifest_path, "r") as f:
            all_records = json.load(f)
            
        self.items = []
        for rec in all_records:
            # [NEW] 直接读取预处理阶段定好的 split 标签
            if rec.get("split") == self.split:
                self.items.append(rec)

        print(f"Loaded {len(self.items)} samples for split: {split}")

        # Rare sampling indices (only for training)
        self.rare_indices = []
        self.common_indices = []
        self.lesion_indices = []     # cases with at least one lesion
        self.no_lesion_indices = []  # cases with no lesion
        if split == "train":
            for idx, rec in enumerate(self.items):
                seq_raw = str(rec.get("sequence_raw", ""))
                if self._is_rare_sequence(seq_raw):
                    self.rare_indices.append(idx)
                else:
                    self.common_indices.append(idx)
                # lesion / no-lesion split
                parts = [p.strip() for p in seq_raw.split(",") if p.strip()]
                has_lesion = len(parts) > 1  # more than just the liver token
                if has_lesion:
                    self.lesion_indices.append(idx)
                else:
                    self.no_lesion_indices.append(idx)

        # 3. Training Augmentation
        self.aug = None
        if split == "train":
            self.aug = Augmentor3D()

        # RAM 缓存已禁用，数据在 __getitem__ 中按需从磁盘读取

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        cache_path = rec["cache_path"]
        # Resolve relative paths against the manifest's directory so the package
        # runs correctly from any working directory or server.
        # The preprocess script stores paths as "./data_cache/summary/<id>.npz"
        # (relative to the original project root); canonicalize to absolute here.
        if not os.path.isabs(cache_path):
            cache_path = os.path.join(self.cache_dir, os.path.basename(cache_path))
        data = np.load(cache_path)
        x = torch.from_numpy(data['x'])
        y = torch.from_numpy(data['y']).long()

        # Augmentation
        if self.aug is not None:
            x = self.aug(x)

        # Lesion order permutation: shuffle lesion order for multi-lesion training samples
        # Lesion order is semantically arbitrary, so any permutation is a valid new sample
        if self.split == "train":
            y = self._shuffle_lesion_order(y)

        rid = str(self.items[idx]["id"])
        return x, y, rid

    @staticmethod
    def _shuffle_lesion_order(y: torch.Tensor) -> torch.Tensor:
        """Randomly permute lesion order in the sequence.
        Sequence format: [BOS, liver, t1, p1, q1, t2, p2, q2, ..., EOS]
        """
        tokens = y.tolist()
        # Find EOS position
        try:
            from .config import EOS, BOS
            eos_pos = tokens.index(EOS)
        except ValueError:
            return y  # no EOS found, return as-is

        # body = everything between liver token and EOS: [t1,p1,q1, t2,p2,q2, ...]
        # tokens[0]=BOS, tokens[1]=liver, tokens[2..eos_pos-1]=lesion tokens, tokens[eos_pos]=EOS
        header = tokens[:2]          # [BOS, liver]
        body   = tokens[2:eos_pos]   # lesion triplets
        tail   = tokens[eos_pos:]    # [EOS] (+ any padding)

        n_lesions = len(body) // 3
        if n_lesions < 2:
            return y  # nothing to permute

        lesions = [body[i*3:(i+1)*3] for i in range(n_lesions)]
        random.shuffle(lesions)
        new_tokens = header + [t for l in lesions for t in l] + tail
        return torch.tensor(new_tokens, dtype=torch.long)

    @staticmethod
    def _is_rare_sequence(seq_raw: str) -> bool:
        parts = [p.strip() for p in seq_raw.split(",") if p.strip()]
        if not parts:
            return False
        rest = parts[1:]
        for i in range(0, len(rest), 3):
            if i + 2 >= len(rest):
                break
            t_raw, p_raw, q_raw = rest[i], rest[i + 1], rest[i + 2]
            t = t_raw.strip().upper()
            p = p_raw.strip().upper()
            q = q_raw.strip().lower()

            if t == "HEM":
                return True
            if p in {"CAUDATE", "L_DIFFUSE"}:
                return True

            digits = "".join([c for c in q if c.isdigit()])
            if digits:
                try:
                    if int(digits) >= 3:
                        return True
                except Exception:
                    pass
        return False


class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_ratio=0.4):
        self.primary_indices = list(primary_indices)
        self.secondary_indices = list(secondary_indices)
        self.batch_size = batch_size
        self.secondary_batch_size = int(round(batch_size * secondary_ratio))
        self.primary_batch_size = batch_size - self.secondary_batch_size

        if self.secondary_batch_size <= 0 or self.primary_batch_size <= 0:
            raise ValueError("Invalid secondary_ratio for given batch_size.")

    def __iter__(self):
        primary = self.primary_indices[:]
        secondary = self.secondary_indices[:]
        random.shuffle(primary)
        random.shuffle(secondary)

        # Cycle secondary if too small
        sec_iter = iter(secondary)

        def next_secondary(k):
            nonlocal sec_iter, secondary
            out = []
            while len(out) < k:
                try:
                    out.append(next(sec_iter))
                except StopIteration:
                    random.shuffle(secondary)
                    sec_iter = iter(secondary)
            return out

        # Yield batches
        for i in range(0, len(primary), self.primary_batch_size):
            p_batch = primary[i:i + self.primary_batch_size]
            if len(p_batch) < self.primary_batch_size:
                break
            s_batch = next_secondary(self.secondary_batch_size) if secondary else []
            batch = p_batch + s_batch
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def collate_fn(batch):
    xs, ys, rids = zip(*batch)
    xs = torch.stack(xs)
    
    # Pad sequences
    max_len = max([len(y) for y in ys])
    ys_padded = torch.full((len(ys), max_len), PAD, dtype=torch.long)
    for i, y in enumerate(ys):
        ys_padded[i, :len(y)] = y
        
    return xs, ys_padded, rids
