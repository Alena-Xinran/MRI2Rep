"""
GradCAM Visualizer for MRI2Rep
================================
Uses gradients of each predicted token's logit w.r.t. the backbone feature maps
(24×24×12 before vis_pool) to produce spatial attribution maps.

Usage
-----
  python gradcam_vis.py --run_dir runs/exp_YYYYMMDD_HHMMSS
  python gradcam_vis.py --run_dir runs/... --case_id Liv_XXXXXXX
  python gradcam_vis.py --run_dir runs/... --n_cases 5 --split test

Outputs (inside <run_dir>/gradcam_maps/)
  <case_id>/gradcam_step<N>_<token>.png   — per-token GradCAM (axial/coronal/sagittal × ART/PV)
  <case_id>/gradcam_summary.png           — all tokens in a grid (axial mid-slice)
"""

import os, sys, math, json, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    Config, PAD, BOS, EOS, NO_LESION,
    LIVER_TYPES, TUMOR_TYPES, POSITIONS, QUANTITIES,
    OFFSET_LIVER, OFFSET_TUMOR, OFFSET_POS, OFFSET_QTY, VOCAB_SIZE,
)
from src.dataset import MRISeqDataset
from src.model import MRIReportGenerator
from src.engine import parse_sequence
from src.utils import set_seed


def token_name(tid):
    if tid == PAD:       return "PAD"
    if tid == BOS:       return "BOS"
    if tid == EOS:       return "EOS"
    if tid == NO_LESION: return "NO_LESION"
    if OFFSET_LIVER <= tid < OFFSET_TUMOR: return LIVER_TYPES[tid - OFFSET_LIVER]
    if OFFSET_TUMOR <= tid < OFFSET_POS:   return TUMOR_TYPES[tid - OFFSET_TUMOR]
    if OFFSET_POS   <= tid < OFFSET_QTY:   return POSITIONS[tid - OFFSET_POS]
    if OFFSET_QTY   <= tid < VOCAB_SIZE:   return QUANTITIES[tid - OFFSET_QTY]
    return f"tok{tid}"


# ── GradCAM core ──────────────────────────────────────────────────────────────

class GradCAMExtractor:
    """
    Hooks onto model.backbone to capture forward feature maps and backward gradients.
    Feature map shape: (B, d_model, 24, 24, 12)  [before vis_pool]
    """
    def __init__(self, model: MRIReportGenerator):
        self.model = model
        self._feat = None
        self._grad = None
        self._hook_f = model.backbone.register_forward_hook(self._save_feat)
        self._hook_b = model.backbone.register_full_backward_hook(self._save_grad)

    def _save_feat(self, module, inp, out):
        self._feat = out  # (B, C, H, W, D)

    def _save_grad(self, module, grad_in, grad_out):
        self._grad = grad_out[0]  # (B, C, H, W, D)

    def remove(self):
        self._hook_f.remove()
        self._hook_b.remove()

    def compute(self, img: torch.Tensor, token_id: int, dec_input: torch.Tensor) -> np.ndarray:
        """
        Forward + backward for one token step.
        Returns GradCAM volume (H, W, D) normalised to [0,1].
        img       : (1, 3, H, W, D)
        dec_input : (1, L) token ids fed to decoder so far (including BOS)
        token_id  : the predicted token whose score we differentiate
        """
        self.model.zero_grad()
        logits, *_ = self.model(img, dec_input)   # (1, L, vocab)
        # Score = logit of the last position for the predicted token
        score = logits[0, -1, token_id]
        score.backward()

        grad = self._grad   # (1, C, H, W, D)
        feat = self._feat   # (1, C, H, W, D)

        # Global-average-pool gradients over spatial dims → channel weights
        weights = grad.mean(dim=(2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)
        cam = (weights * feat).sum(dim=1, keepdim=True)    # (1, 1, H, W, D)
        cam = F.relu(cam)

        # Upsample to full volume size
        H_full = img.shape[2]
        W_full = img.shape[3]
        D_full = img.shape[4]
        cam = F.interpolate(
            cam.permute(0, 1, 4, 2, 3),   # → (1,1,D,H,W) for interpolate
            size=(D_full, H_full, W_full),
            mode='trilinear', align_corners=True
        ).permute(0, 1, 3, 4, 2)           # → (1,1,H,W,D)

        vol = cam.squeeze().detach().cpu().float().numpy()   # (H, W, D)
        vmin, vmax = vol.min(), vol.max()
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
        return vol


# ── Greedy decode collecting GradCAM per step ─────────────────────────────────

def decode_with_gradcam(model, extractor, img, max_len=28):
    """
    Returns:
        tokens     : list[int]  generated token IDs
        step_cams  : list[ndarray(H,W,D)]  one GradCAM per predicted token (after BOS)
    """
    model.eval()
    device = img.device
    tokens = [BOS]
    step_cams = []

    # First pass: greedy decode without grad to get the full sequence
    with torch.no_grad():
        memory, _ = model.encode_image(img)
        dec_in = torch.tensor([[BOS]], device=device)
        for _ in range(max_len):
            tgt = model.tok_emb(dec_in) * math.sqrt(model.cfg.d_model)
            tgt = model.pos_emb(tgt)
            L = tgt.size(1)
            mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
            out = model.decoder(tgt, memory, tgt_mask=mask)
            if model.decoder.norm is not None:
                out = model.decoder.norm(out)
            next_tok = int(model.head(out[0, -1]).argmax().item())
            tokens.append(next_tok)
            dec_in = torch.cat([dec_in, torch.tensor([[next_tok]], device=device)], dim=1)
            if next_tok == EOS:
                break

    # Second pass: GradCAM for each step (re-run forward with grad)
    for step in range(1, len(tokens)):
        tok_id = tokens[step]
        if tok_id in (EOS, PAD):
            break
        dec_input = torch.tensor([tokens[:step]], device=device)   # BOS … prev_tok
        cam = extractor.compute(img, tok_id, dec_input)
        step_cams.append(cam)

    return tokens, step_cams


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_gradcam(mri_vol, cam_vol, token_label, save_path, alpha=0.5):
    """3×2 figure: axial/coronal/sagittal × ART/PV with GradCAM overlay."""
    C, H, W, D = mri_vol.shape
    mid_h, mid_w, mid_d = H // 2, W // 2, D // 2
    # Mask GradCAM to liver region only
    liver_mask = (mri_vol[2] > 0.5).astype(np.float32)
    cam_vol = cam_vol * liver_mask

    slice_configs = [
        ("Axial",    mri_vol[:, :, :, mid_d], cam_vol[:, :, mid_d]),
        ("Coronal",  mri_vol[:, mid_h, :, :], cam_vol[mid_h, :, :]),
        ("Sagittal", mri_vol[:, :, mid_w, :], cam_vol[:, mid_w, :]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"GradCAM  |  Token: {token_label}", fontsize=13, fontweight="bold")

    for col, (view, mri_sl, cam_sl) in enumerate(slice_configs):
        for row, ch in enumerate([0, 1]):
            ax = axes[row, col]
            img_sl = mri_sl[ch]
            if col in (1, 2):
                img_sl = img_sl.T
                cam_sl_disp = cam_sl.T
            else:
                cam_sl_disp = cam_sl
            ax.imshow(img_sl, cmap="gray", vmin=0, vmax=1, origin="upper")
            ax.imshow(cam_sl_disp, cmap="jet", alpha=alpha, vmin=0, vmax=1, origin="upper")
            if row == 0:
                ax.set_title(view, fontsize=10)
            ax.set_ylabel("ART" if ch == 0 else "PV", fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_summary(mri_vol, step_cams, token_ids, save_path, alpha=0.45):
    """Compact grid: one column per token, axial mid-slice, ART+PV rows."""
    C, H, W, D = mri_vol.shape
    mid_d = D // 2
    liver_mask = (mri_vol[2] > 0.5).astype(np.float32)
    step_cams = [cam * liver_mask for cam in step_cams]
    n = len(step_cams)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle("GradCAM Summary (Axial Mid-Slice)", fontsize=12)

    for i, (cam, tid) in enumerate(zip(step_cams, token_ids)):
        cam_ax = cam[:, :, mid_d]
        for row, ch in enumerate([0, 1]):
            ax = axes[row, i]
            ax.imshow(mri_vol[ch, :, :, mid_d], cmap="gray", vmin=0, vmax=1, origin="upper")
            ax.imshow(cam_ax, cmap="jet", alpha=alpha, vmin=0, vmax=1, origin="upper")
            ax.axis("off")
            if row == 0:
                ax.set_title(token_name(tid), fontsize=7, rotation=30, ha="left")
    axes[0, 0].set_ylabel("ART", fontsize=9)
    axes[1, 0].set_ylabel("PV", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",  required=True)
    parser.add_argument("--ckpt",     default="best_lesion1p_model.pth")
    parser.add_argument("--split",    default="test", choices=["test", "val", "train"])
    parser.add_argument("--case_id",  default=None)
    parser.add_argument("--n_cases",  default=1, type=int)
    parser.add_argument("--seed",     default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = Config()
    run_cfg_path = os.path.join(args.run_dir, "config.json")
    if os.path.exists(run_cfg_path):
        with open(run_cfg_path) as f:
            for k, v in json.load(f).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        print(f"Loaded config from {run_cfg_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt_path = os.path.join(args.run_dir, args.ckpt)
    weights = torch.load(ckpt_path, map_location=device)
    if "model" in weights:
        weights = weights["model"]
    model = MRIReportGenerator(cfg).to(device)
    model.load_state_dict(weights, strict=True)
    print(f"Loaded {args.ckpt}")

    # Dataset
    ds = MRISeqDataset(cfg, split=args.split)
    if args.case_id is not None:
        indices = [i for i, r in enumerate(ds.items) if str(r["id"]) == args.case_id]
        if not indices:
            print(f"case_id {args.case_id} not found"); sys.exit(1)
    else:
        all_idx = list(range(len(ds)))
        random.shuffle(all_idx)
        indices = all_idx[:args.n_cases]

    print(f"Visualizing {len(indices)} case(s) from {args.split} split")
    out_root = os.path.join(args.run_dir, "gradcam_maps")
    extractor = GradCAMExtractor(model)

    for idx in indices:
        rec = ds.items[idx]
        case_id = str(rec["id"])
        print(f"\n── {case_id} ──")

        cache_path = rec["cache_path"]
        if not os.path.isabs(cache_path):
            cache_path = os.path.join(ds.cache_dir, os.path.basename(cache_path))
        data = np.load(cache_path)
        x_np = data["x"]   # (3, H, W, D)
        y_np = data["y"]

        x = torch.from_numpy(x_np).unsqueeze(0).to(device)

        tokens, step_cams = decode_with_gradcam(model, extractor, x)
        pred_tids = tokens[1:]  # skip BOS

        print(f"  GT  : {[token_name(t) for t in y_np.tolist() if t != PAD]}")
        print(f"  Pred: {[token_name(t) for t in tokens if t != PAD]}")

        case_out = os.path.join(out_root, case_id)
        os.makedirs(case_out, exist_ok=True)

        for i, (cam, tid) in enumerate(zip(step_cams, pred_tids)):
            if tid in (EOS, PAD):
                break
            save_path = os.path.join(case_out, f"gradcam_step{i:02d}_{token_name(tid)}.png")
            plot_gradcam(x_np, cam, token_name(tid), save_path)
            print(f"  Saved: {save_path}")

        plot_summary(x_np, step_cams[:len(pred_tids)],
                     pred_tids[:len(step_cams)],
                     os.path.join(case_out, "gradcam_summary.png"))
        print(f"  Saved summary")

    extractor.remove()
    print(f"\nAll GradCAM maps saved to: {out_root}")


if __name__ == "__main__":
    main()
