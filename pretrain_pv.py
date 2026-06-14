"""
ART → PV Pretraining
=====================
Self-supervised pretraining of the visual backbone on the target-domain data.

Task:  given [ART, 0, MASK] as input, reconstruct the PV channel.
Why:   forces the backbone to learn arterial-vs-portal enhancement patterns
       (APHE, washout, etc.) without requiring any lesion-level annotations.

Architecture:
    Encoder : ConvBackbone3D (identical to MRIReportGenerator.backbone, in_ch=3)
    Decoder : symmetric transposed-conv upsampler → (B, 1, 192, 192, 96)
    Loss    : 0.8 * MSE + 0.2 * (1 - SSIM) within liver mask

Output:
    ./pretrained/backbone_pv_pretrain.pth   ← backbone-only state dict

Usage:
    python pretrain_pv.py
    python main.py                          ← auto-loads pretrained backbone
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import Config
from src.model import ConvBackbone3D


def ssim_loss(pred: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor, win: int = 7) -> torch.Tensor:
    """
    1 - SSIM, computed locally with avg_pool3d as the window mean estimator.
    Only masked (liver) voxels contribute to the final value.
    pred/target: (B, 1, H, W, D), mask: (B, 1, H, W, D)
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = win // 2

    def lmean(x):
        return torch.nn.functional.avg_pool3d(
            x, kernel_size=win, stride=1, padding=pad)

    mu_p  = lmean(pred)
    mu_t  = lmean(target)
    mu_pp = mu_p ** 2
    mu_tt = mu_t ** 2
    mu_pt = mu_p * mu_t

    sig_pp = lmean(pred   ** 2) - mu_pp
    sig_tt = lmean(target ** 2) - mu_tt
    sig_pt = lmean(pred * target) - mu_pt

    num  = (2 * mu_pt + C1) * (2 * sig_pt + C2)
    den  = (mu_pp + mu_tt + C1) * (sig_pp + sig_tt + C2)
    smap = 1.0 - num / (den + 1e-8)          # (B, 1, H, W, D), lower = better

    return (smap * mask).sum() / (mask.sum() + 1e-6)


# ── Decoder ───────────────────────────────────────────────────────────────────

class ReconDecoder3D(nn.Module):
    """
    (B, d_model, 24, 24, 12) → (B, 1, 192, 192, 96)
    Three stride-2 transposed convolutions, symmetric to the encoder.
    Intermediate channels scale proportionally with in_ch so that the decoder
    capacity matches the encoder regardless of d_model (128 or 256).
    """
    def __init__(self, in_ch: int = 128):
        super().__init__()
        mid1 = max(64, in_ch // 2)   # e.g. 128→64 or 256→128
        mid2 = max(32, in_ch // 4)   # e.g. 128→32 or 256→64
        self.net = nn.Sequential(
            nn.ConvTranspose3d(in_ch, mid1, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(mid1), nn.GELU(),
            nn.ConvTranspose3d(mid1, mid2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(mid2), nn.GELU(),
            nn.ConvTranspose3d(mid2,    1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),   # PV is normalised to [0, 1] in the cache
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Full pretrain model ───────────────────────────────────────────────────────

class PVPretrainModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        # Same backbone as MRIReportGenerator — weights transfer directly
        self.backbone = ConvBackbone3D(in_ch=3, base=cfg.backbone_base, out_ch=cfg.d_model)
        self.decoder  = ReconDecoder3D(in_ch=cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 3, H, W, D) with PV channel zeroed out"""
        feats   = self.backbone(x)   # (B, d_model, H/8, W/8, D/8)
        pv_pred = self.decoder(feats) # (B, 1, H, W, D)
        return pv_pred


# ── Dataset ───────────────────────────────────────────────────────────────────

class PVPretrainDataset(Dataset):
    """
    Reuses the existing .npz cache (no extra preprocessing).
    Uses train + val splits — no label leakage because the task is purely visual.
    """
    def __init__(self, cfg: Config):
        self.cache_dir = os.path.join(cfg.cache_root, cfg.cache_tag)
        manifest_path  = os.path.join(self.cache_dir, "manifest.json")
        with open(manifest_path) as f:
            records = json.load(f)
        self.items = [r for r in records if r.get("split") in ("train", "val", "test")]
        print(f"[Pretrain] Dataset: {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cache_path = self.items[idx]["cache_path"]
        if not os.path.isabs(cache_path):
            cache_path = os.path.join(self.cache_dir, os.path.basename(cache_path))
        data = np.load(cache_path)
        x    = torch.from_numpy(data['x'])   # (3, H, W, D) — [ART, PV, MASK]

        pv_gt    = x[1:2].clone()   # (1, H, W, D)  reconstruction target
        mask     = x[2:3].clone()   # (1, H, W, D)  liver mask

        x_in     = x.clone()
        x_in[1]  = 0.0              # zero-out PV channel

        return x_in, pv_gt, mask


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = "./pretrained"
    os.makedirs(out_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = PVPretrainDataset(cfg)
    # GPU VRAM: batch=48 uses ~9 GB (fits alongside main training).
    # System RAM: num_workers=2, prefetch_factor=1 keeps prefetch at ~4 GB.
    pretrain_batch_size = 48
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    num_workers = max(0, min(slurm_cpus - 1, 2)) if slurm_cpus > 0 else 2
    loader  = DataLoader(
        dataset,
        batch_size  = pretrain_batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        persistent_workers = (num_workers > 0),
        prefetch_factor    = 1 if num_workers > 0 else None,
    )
    print(f"Pretrain DataLoader: batch={pretrain_batch_size}, num_workers={num_workers}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = PVPretrainModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Training ──────────────────────────────────────────────────────────────
    best_loss = float("inf")
    history   = []

    print("Starting ART → PV pretraining (50 epochs) …")
    for epoch in range(1, 51):
        model.train()
        running = 0.0

        for x_in, pv_gt, mask in loader:
            x_in  = x_in.to(device)
            pv_gt = pv_gt.to(device)
            mask  = mask.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                pv_pred  = model(x_in)
                diff     = (pv_pred - pv_gt) ** 2
                mse      = (diff * mask).sum() / (mask.sum() + 1e-6)
                loss     = 0.8 * mse + 0.2 * ssim_loss(pv_pred, pv_gt, mask)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            running += loss.item()

        epoch_loss = running / len(loader)
        history.append(epoch_loss)
        scheduler.step()
        print(f"Epoch {epoch:>3}/50 | loss={epoch_loss:.6f}")

        # Save best backbone weights only
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # Strip torch.compile prefix (_orig_mod.) and keep only backbone keys
            raw_state = {
                k.replace("_orig_mod.", ""): v
                for k, v in model.state_dict().items()
            }
            backbone_state = {
                k[len("backbone."):]: v
                for k, v in raw_state.items()
                if k.startswith("backbone.")
            }
            torch.save(backbone_state, os.path.join(out_dir, "backbone_pv_pretrain.pth"))
            print(f"  >>> Best backbone saved (loss={best_loss:.6f})")

        # Loss curve every 10 epochs
        if epoch % 10 == 0:
            plt.figure(figsize=(8, 4))
            plt.plot(history, marker=".")
            plt.title("ART → PV Pretrain Loss (MSE within mask)")
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "pretrain_loss.png"))
            plt.close()

    print(f"\nPretraining complete.  Best loss: {best_loss:.6f}")
    print(f"Backbone weights → {out_dir}/backbone_pv_pretrain.pth")


if __name__ == "__main__":
    main()
