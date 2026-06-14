import os
import sys
import signal
import torch
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import asdict

from src.config import Config, PAD, OFFSET_TUMOR, OFFSET_POS, OFFSET_QTY, VOCAB_SIZE, TUMOR_TYPES, POSITIONS, LIVER_TYPES, OFFSET_LIVER
from src.dataset import MRISeqDataset, collate_fn, TwoStreamBatchSampler
from src.model import MRIReportGenerator
from src.engine import train_one_epoch, evaluate
from src.utils import set_seed
from torch.utils.data import DataLoader

# ── Graceful shutdown on SIGTERM / SIGINT ─────────────────────────────────────
# When a job scheduler (SLURM, PBS, …) sends SIGTERM, or Ctrl-C is pressed,
# the flag is set and training finishes the current epoch cleanly before saving
# the checkpoint and exiting.  The next run with --resume picks up exactly here.
_shutdown_requested = False

def _handle_signal(signum, frame):
    global _shutdown_requested
    print(f"\n[SIGTERM] Will checkpoint after current epoch and exit gracefully.")
    _shutdown_requested = True

# SIGTERM: graceful — waits for epoch to finish, then saves and exits
signal.signal(signal.SIGTERM, _handle_signal)
# SIGINT (Ctrl+C): keep default behavior → raises KeyboardInterrupt immediately,
# caught in the training loop below → saves last completed checkpoint and exits
# ──────────────────────────────────────────────────────────────────────────────


def save_config(cfg: Config, run_dir: str):
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=4)
    print(f"Configuration saved to: {config_path}")


def update_plots(log_file: str, run_dir: str):
    try:
        df = pd.read_csv(log_file)
        if len(df) < 2:
            return

        epochs = df['epoch']

        # Loss
        plt.figure(figsize=(10, 6))
        if 'train_seq_nll' in df.columns:
            plt.plot(epochs, df['train_seq_nll'], label='Train Seq NLL', marker='.')
        elif 'train_nll' in df.columns:
            plt.plot(epochs, df['train_nll'], label='Train NLL', marker='.')
        plt.plot(epochs, df['val_nll'], label='Val NLL', marker='.')

        plt.title('Negative Log Likelihood (Lower is Better)')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(run_dir, "curve_loss.png")); plt.close()

        # Pair / Triplet F1
        plt.figure(figsize=(10, 6))
        for col, label, color, ls in [
            ('val_pair_f1',        'Val Pair F1',          'green', '-'),
            ('val_pair_f1_micro',  'Val Pair F1 (Micro)',   'green', '--'),
            ('val_triplet_f1',     'Val Triplet F1',        'teal',  '-'),
            ('val_triplet_f1_micro','Val Triplet F1 (Micro)','teal', '--'),
        ]:
            if col in df.columns:
                plt.plot(epochs, df[col], label=label, color=color, linestyle=ls, marker='.')
        plt.title('Lesion F1 (Pair/Triplet)')
        plt.xlabel('Epoch'); plt.ylabel('F1 Score'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(run_dir, "curve_f1.png")); plt.close()

        # Liver acc
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, df['val_liver_acc'], label='Val Liver Acc', color='orange', marker='.')
        plt.title('Liver Type Accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(run_dir, "curve_acc.png")); plt.close()

        # Pair F1 solo
        if 'val_pair_f1' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, df['val_pair_f1'], label='Val Pair F1', color='green', marker='.')
            plt.title('Pair F1')
            plt.xlabel('Epoch'); plt.ylabel('F1 Score'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(run_dir, "curve_pair_f1.png")); plt.close()

        # Triplet F1 solo
        if 'val_triplet_f1' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, df['val_triplet_f1'], label='Val Triplet F1', color='teal', marker='.')
            plt.title('Triplet F1')
            plt.xlabel('Epoch'); plt.ylabel('F1 Score'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(run_dir, "curve_triplet_f1.png")); plt.close()

        # Qty accuracy
        if 'val_qty_acc' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, df['val_qty_acc'], label='Val Qty Acc', color='purple', marker='.')
            plt.title('Quantity Accuracy | (Type, Position) correct')
            plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(run_dir, "curve_qty_acc.png")); plt.close()

        # No-lesion accuracy
        if 'val_no_lesion_acc' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, df['val_no_lesion_acc'], label='Val No-Lesion Acc', color='brown', marker='.')
            plt.title('No-Lesion Accuracy')
            plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(run_dir, "curve_no_lesion_acc.png")); plt.close()

        # Macro F1
        if 'val_type_macro_f1' in df.columns or 'val_pos_macro_f1' in df.columns:
            plt.figure(figsize=(10, 6))
            if 'val_type_macro_f1' in df.columns:
                plt.plot(epochs, df['val_type_macro_f1'], label='Val Type Macro F1', color='blue', marker='.')
            if 'val_pos_macro_f1' in df.columns:
                plt.plot(epochs, df['val_pos_macro_f1'], label='Val Pos Macro F1', color='red', marker='.')
            plt.title('Macro F1')
            plt.xlabel('Epoch'); plt.ylabel('F1 Score'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(run_dir, "curve_macro_f1.png")); plt.close()

    except Exception as e:
        print(f"[Warning] Plotting failed: {e}")


def build_lesion_sampler(train_ds, cfg):
    """
    Build a TwoStreamBatchSampler that:
      - Primary stream  : all lesion cases with rare types *oversampled*
                          (restores the rare_sampling_ratio logic that was lost)
      - Secondary stream: no-lesion cases capped at no_lesion_ratio per batch

    Returns (sampler, description_string).
    """
    if not train_ds.lesion_indices:
        return None, "No lesion samples found — using plain DataLoader."

    if train_ds.rare_indices and train_ds.common_indices:
        # Inflate rare cases so they make up ~rare_sampling_ratio of the lesion pool.
        # E.g. rare_ratio=0.4 → target_rare = 0.4/0.6 * len(common) items from rare pool.
        ratio = cfg.rare_sampling_ratio
        target_rare = max(len(train_ds.rare_indices),
                          int(len(train_ds.common_indices) * ratio / max(1e-6, 1.0 - ratio)))
        repeats     = max(1, (target_rare + len(train_ds.rare_indices) - 1)
                             // len(train_ds.rare_indices))
        inflated    = (train_ds.rare_indices * repeats)[:target_rare]
        lesion_pool = train_ds.common_indices + inflated
        desc = (f"3-stream sampling: {len(train_ds.common_indices)} common + "
                f"{len(inflated)} inflated-rare (x{repeats} from {len(train_ds.rare_indices)}) "
                f"+ {len(train_ds.no_lesion_indices)} no-lesion "
                f"(capped {cfg.no_lesion_ratio:.0%}/batch)")
    else:
        lesion_pool = train_ds.lesion_indices
        desc = (f"2-stream sampling: {len(lesion_pool)} lesion "
                f"+ {len(train_ds.no_lesion_indices)} no-lesion "
                f"(capped {cfg.no_lesion_ratio:.0%}/batch)")

    if train_ds.no_lesion_indices:
        sampler = TwoStreamBatchSampler(
            lesion_pool, train_ds.no_lesion_indices,
            cfg.batch_size, secondary_ratio=cfg.no_lesion_ratio
        )
    else:
        # All samples are lesion-positive; fall back to plain shuffle
        sampler = None
        desc += " [no no-lesion samples, using plain shuffle]"

    return sampler, desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a run directory to resume from "
                             "(e.g. runs/exp_20260220_171000)")
    args = parser.parse_args()

    cfg = Config()
    set_seed(cfg.seed)

    # ── 1. Run directory ──────────────────────────────────────────────────────
    if args.resume:
        run_dir = args.resume.rstrip("/")
        if not os.path.exists(run_dir):
            print(f"[Error] Resume directory not found: {run_dir}")
            sys.exit(1)
        print(f"Resuming experiment from: {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir   = os.path.join(cfg.outdir, f"exp_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"Running new experiment in: {run_dir}")

    if not args.resume:
        save_config(cfg, run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Auto-detect available CPUs from SLURM or os.cpu_count()
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    if slurm_cpus > 0:
        cfg.num_workers = max(0, min(slurm_cpus - 1, cfg.num_workers))
    print(f"num_workers: {cfg.num_workers} (SLURM_CPUS_PER_TASK={slurm_cpus})")

    # ── 2. Datasets & dataloaders ─────────────────────────────────────────────
    train_ds = MRISeqDataset(cfg, split="train")
    val_ds   = MRISeqDataset(cfg, split="val")
    print(f"Train samples: {len(train_ds)},  Val samples: {len(val_ds)}")

    batch_sampler, sampler_desc = build_lesion_sampler(train_ds, cfg)
    print(sampler_desc)

    loader_kwargs = dict(
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    if batch_sampler is not None:
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  shuffle=True, **loader_kwargs)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, **loader_kwargs)

    # ── Common pairs: training pairs with freq >= 20 (used for filtered eval) ─
    from collections import Counter as _Ctr
    from src.engine import parse_sequence as _ps
    _pf = _Ctr()
    for _, _y, _ in DataLoader(train_ds, batch_size=256, shuffle=False,
                                num_workers=0, collate_fn=collate_fn):
        for _yi in _y:
            for _t, _p, _ in _ps(_yi.numpy())['lesions']:
                _pf[(_t, _p)] += 1
    common_pairs = {p for p, c in _pf.items() if c >= 20}
    print(f"Common pairs (train_freq>=20): {len(common_pairs)}/45")

    # ── 3. Model, optimiser, scheduler ───────────────────────────────────────
    model = MRIReportGenerator(cfg).to(device)

    # Load pretrained backbone
    pretrained_backbone = cfg.pretrained_backbone
    print(f"d_model={cfg.d_model}, backbone_base={cfg.backbone_base}")
    if os.path.exists(pretrained_backbone):
        backbone_weights = torch.load(pretrained_backbone, map_location=device)
        try:
            missing, unexpected = model.backbone.load_state_dict(backbone_weights, strict=True)
            print(f"Loaded pretrained backbone from {pretrained_backbone} "
                  f"(missing={missing}, unexpected={unexpected})")
        except RuntimeError as e:
            print(f"[Warning] Pretrained backbone shape mismatch — training from scratch.\n  Detail: {e}")
    else:
        print(f"No pretrained backbone found at {pretrained_backbone} — training from scratch.")

    # aux_lesion_head gets 20× LR: it's randomly initialised and needs to learn
    # fast; backbone/decoder are pretrained or large, so they keep the base LR.
    aux_params      = list(model.aux_lesion_head.parameters())
    backbone_params = list(model.backbone.parameters())
    aux_ids         = {id(p) for p in aux_params}
    backbone_ids    = {id(p) for p in backbone_params}
    other_params    = [p for p in model.parameters()
                       if id(p) not in aux_ids and id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.lr * 0.1, "weight_decay": cfg.weight_decay},
        {"params": other_params,    "lr": cfg.lr,       "weight_decay": cfg.weight_decay},
        {"params": aux_params,      "lr": cfg.lr * 20,  "weight_decay": 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-7
    )

    # ── 4. Class-weighted loss ────────────────────────────────────────────────
    weight_tensor = None
    if cfg.use_class_weights:
        counts = np.zeros(VOCAB_SIZE, dtype=np.float64)
        count_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collate_fn
        )
        for _, y, _ in count_loader:
            y_flat = y.view(-1).numpy()
            counts += np.bincount(y_flat, minlength=VOCAB_SIZE)

        def eff_weight(cnt, beta):
            w = (1.0 - beta) / (1.0 - np.power(beta, cnt))
            return np.where(cnt > 0, w, 0.0)

        weights = np.ones(VOCAB_SIZE, dtype=np.float32)
        for lo, hi in [(OFFSET_TUMOR, OFFSET_POS),
                       (OFFSET_POS,   OFFSET_QTY),
                       (OFFSET_QTY,   VOCAB_SIZE)]:
            idx     = np.arange(lo, hi)
            w       = eff_weight(counts[idx], cfg.class_weight_beta)
            nonzero = w > 0
            if nonzero.any():
                w = w * (nonzero.sum() / w[nonzero].sum())
            weights[idx] = w

        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=PAD,
        weight=weight_tensor,
        label_smoothing=cfg.label_smoothing,
    )

    # ── 5. Logging ────────────────────────────────────────────────────────────
    log_file = os.path.join(run_dir, "training_log.csv")
    if not args.resume:
        with open(log_file, "w") as f:
            f.write(
                "epoch,train_seq_nll,train_aux_loss,val_nll,val_liver_acc,val_no_lesion_acc,"
                "val_type_f1,val_type_macro_f1,val_pos_macro_f1,"
                "val_pair_f1,val_pair_f1_micro,"
                "val_triplet_f1,val_triplet_f1_micro,val_soft_triplet_f1,"
                "val_case_sen,val_case_spe,val_type_sen_weighted,val_type_spe_weighted,"
                "val_qty_acc,val_qty_within1_acc,val_qty_ordinal_mae,"
                "val_pair_f1_lesion0,val_pair_f1_lesion1p,val_pair_f1_diffuse,val_pair_f1_focal,"
                "val_triplet_f1_lesion0,val_triplet_f1_lesion1p,val_triplet_f1_diffuse,val_triplet_f1_focal,"
                "val_pair_f1_lesion1p_common\n"
            )

    # ── 6. Mixed precision scaler ─────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler() if cfg.use_amp and device.type == 'cuda' else None
    if scaler:
        print("AMP enabled (fp16 mixed precision)")

    # ── 7. Resume from checkpoint ─────────────────────────────────────────────
    start_epoch        = 1
    best_pair_f1       = 0.0
    best_lesion1p_f1   = 0.0
    no_improve_epochs  = 0

    ckpt_path    = os.path.join(run_dir, "last_checkpoint.pth")
    fallback_path = os.path.join(run_dir, "last_model.pth")

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        # Auto-detect architecture: if checkpoint has no vis_pos_emb, rebuild with enc_layers=0
        if 'vis_pos_emb' not in ckpt["model"] and cfg.enc_layers > 0:
            print(f"[Resume] Checkpoint has no vis_encoder; rebuilding model with enc_layers=0 "
                  f"to match checkpoint architecture (was enc_layers={cfg.enc_layers}).")
            cfg.enc_layers = 0
            model = MRIReportGenerator(cfg).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch       = ckpt["epoch"] + 1
        best_pair_f1      = ckpt["best_pair_f1"]
        best_lesion1p_f1  = ckpt.get("best_lesion1p_f1", 0.0)
        no_improve_epochs = ckpt["no_improve_epochs"]
        if scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        print(f"Resumed from epoch {ckpt['epoch']} | best_pair_f1={best_pair_f1:.4f} | "
              f"no_improve={no_improve_epochs}")
    elif args.resume and os.path.exists(fallback_path):
        # Fallback: only model weights (no optimizer / scheduler state)
        weights = torch.load(fallback_path, map_location=device)
        # Auto-detect architecture: if checkpoint has no vis_pos_emb, rebuild with enc_layers=0
        if 'vis_pos_emb' not in weights and cfg.enc_layers > 0:
            print(f"[Resume] Checkpoint has no vis_encoder; rebuilding model with enc_layers=0 "
                  f"to match checkpoint architecture (was enc_layers={cfg.enc_layers}).")
            cfg.enc_layers = 0
            model = MRIReportGenerator(cfg).to(device)
        model.load_state_dict(weights, strict=True)
        if os.path.exists(log_file):
            df_log = pd.read_csv(log_file)
            if len(df_log) > 0:
                start_epoch   = int(df_log["epoch"].iloc[-1]) + 1
                best_pair_f1  = float(df_log["val_pair_f1"].max())
                print(f"[Fallback] model weights only → epoch {start_epoch}, "
                      f"best_pair_f1={best_pair_f1:.4f}")
                print("[Warning] optimizer/scheduler states not restored — LR schedule restarts.")
                for _ in range(start_epoch - 1):
                    scheduler.step()
                print(f"[Fallback] Scheduler fast-forwarded {start_epoch - 1} steps, "
                      f"LR = {scheduler.get_last_lr()}")
    elif args.resume:
        print("[Warning] No checkpoint found in resume dir; starting from scratch.")

    plot_interval = 5

    # ── 7. Training loop ──────────────────────────────────────────────────────
    # KeyboardInterrupt (Ctrl+C) breaks out immediately mid-epoch.
    # The last *completed* epoch's checkpoint (last_checkpoint.pth) is always safe.
    try:
      for epoch in range(start_epoch, cfg.epochs + 1):

        # Scheduled sampling: only activates after ss_start_epoch
        if (cfg.ss_max_prob > 0
                and epoch > cfg.ss_start_epoch
                and cfg.ss_warmup_epochs > 0):
            effective = epoch - cfg.ss_start_epoch
            ss_prob   = min(cfg.ss_max_prob,
                            cfg.ss_max_prob * effective / cfg.ss_warmup_epochs)
        else:
            ss_prob = 0.0

        # Train
        train_seq_nll, train_aux_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            loss_fn=loss_fn,
            ss_prob=ss_prob,
            word_dropout_p=cfg.word_dropout_p,
            scaler=scaler,
            aux_weight=cfg.aux_weight,
        )

        # Validate
        metrics = evaluate(model, val_loader, device, common_pairs=common_pairs)

        print(
            f"Epoch {epoch:>3} | "
            f"TrainNLL {train_seq_nll:.4f}  AuxLoss {train_aux_loss:.4f} | "
            f"ValNLL {metrics['nll']:.4f} | "
            f"LiverAcc {metrics['liver_acc']:.3f} | "
            f"NoLesAcc {metrics['no_lesion_acc']:.3f} | "
            f"PairF1 {metrics['pair_f1']:.3f}/{metrics['pair_f1_micro']:.3f} | "
            f"TripF1 {metrics['triplet_f1']:.3f}  SoftTrip {metrics['soft_triplet_f1']:.3f} | "
            f"QtyExact {metrics['qty_acc']:.3f}  Qty±1 {metrics['qty_within1_acc']:.3f}  "
            f"QtyMAE {metrics['qty_ordinal_mae']:.3f} | "
            f"CaseSen {metrics['case_sen']:.3f}  CaseSpe {metrics['case_spe']:.3f} | "
            f"TypeSen(w) {metrics['type_sen_weighted']:.3f}  TypeSpe(w) {metrics['type_spe_weighted']:.3f}  "
            f"PairF1_common {metrics['pair_f1_lesion1p_common']:.3f}"
            + (f"  [SS={ss_prob:.2f}]" if ss_prob > 0 else "")
        )

        scheduler.step()

        # Log
        with open(log_file, "a") as f:
            f.write(
                f"{epoch},{train_seq_nll},{train_aux_loss},{metrics['nll']},"
                f"{metrics['liver_acc']},{metrics['no_lesion_acc']},"
                f"{metrics['type_f1']},{metrics['type_macro_f1']},{metrics['pos_macro_f1']},"
                f"{metrics['pair_f1']},{metrics['pair_f1_micro']},"
                f"{metrics['triplet_f1']},{metrics['triplet_f1_micro']},{metrics['soft_triplet_f1']},"
                f"{metrics['case_sen']},{metrics['case_spe']},"
                f"{metrics['type_sen_weighted']},{metrics['type_spe_weighted']},"
                f"{metrics['qty_acc']},{metrics['qty_within1_acc']},{metrics['qty_ordinal_mae']},"
                f"{metrics['pair_f1_lesion0']},{metrics['pair_f1_lesion1p']},"
                f"{metrics['pair_f1_diffuse']},{metrics['pair_f1_focal']},"
                f"{metrics['triplet_f1_lesion0']},{metrics['triplet_f1_lesion1p']},"
                f"{metrics['triplet_f1_diffuse']},{metrics['triplet_f1_focal']},"
                f"{metrics['pair_f1_lesion1p_common']}\n"
            )

        if epoch % plot_interval == 0:
            update_plots(log_file, run_dir)

        # ── Best-model checkpoint ─────────────────────────────────────────────
        # Guard: don't save in early epochs where the trivial "predict no-lesion"
        # strategy yields a misleadingly high pair_f1 (~0.37 = no-lesion prevalence).
        is_qualified = (
            epoch >= cfg.min_epoch_best
            and metrics['pair_f1_lesion1p'] >= cfg.min_lesion1p_f1_for_best
        )
        improved = False
        # Primary criterion: pair_f1_lesion1p (lesion-positive cases only).
        # No-lesion cases are excluded — predicting "no lesion" trivially achieves
        # high pair_f1/pair_f1_micro but gives zero signal on actual detection quality.
        if is_qualified and metrics['pair_f1_lesion1p'] > best_lesion1p_f1:
            best_lesion1p_f1 = metrics['pair_f1_lesion1p']
            best_pair_f1     = metrics['pair_f1_lesion1p']   # keep in sync for ckpt
            improved = True
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            torch.save(model.state_dict(), os.path.join(run_dir, "best_lesion1p_model.pth"))
            print(f"  >>> Best Lesion1p Pair F1 = {best_lesion1p_f1:.4f}  (saved best_model.pth)")
        if improved:
            no_improve_epochs = 0
        elif epoch >= cfg.min_epoch_best:
            no_improve_epochs += 1

        # Early stopping (counts from any non-improving epoch, including pre-qualified ones)
        if cfg.early_stop_patience > 0 and no_improve_epochs >= cfg.early_stop_patience:
            print(f"Early stopping at epoch {epoch}: "
                  f"val pair_f1 not improved for {cfg.early_stop_patience} epochs.")
            update_plots(log_file, run_dir)
            break


        # ── Full checkpoint (always saved — enables resume after any interruption) ──
        torch.save({
            "epoch":             epoch,
            "model":             model.state_dict(),
            "optimizer":         optimizer.state_dict(),
            "scheduler":         scheduler.state_dict(),
            "scaler":            scaler.state_dict() if scaler else None,
            "best_pair_f1":      best_pair_f1,
            "best_lesion1p_f1":  best_lesion1p_f1,
            "no_improve_epochs": no_improve_epochs,
        }, os.path.join(run_dir, "last_checkpoint.pth"))
        # Plain weights file for inference / inspection without optimizer state
        torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pth"))

        # ── Graceful shutdown requested (SIGTERM) ─────────────────────────────
        if _shutdown_requested:
            print(f"[SIGTERM] Checkpoint saved at epoch {epoch}.  Exiting.")
            update_plots(log_file, run_dir)
            break

    except KeyboardInterrupt:
        # Ctrl+C: interrupted mid-epoch.  last_checkpoint.pth already holds the
        # last *completed* epoch — just exit cleanly without corrupting it.
        print(f"\n[Ctrl+C] Interrupted mid-epoch.  "
              f"Checkpoint of last completed epoch is safe in {run_dir}/last_checkpoint.pth")
        print(f"Resume with:  python main.py --resume {run_dir}")
        update_plots(log_file, run_dir)
        return

    print("Training complete.")
    update_plots(log_file, run_dir)


if __name__ == "__main__":
    main()
