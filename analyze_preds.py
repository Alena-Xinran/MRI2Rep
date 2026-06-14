"""
Val set inference: compare GT vs model predictions case by case.
Usage: python analyze_preds.py [--exp runs/exp_xxx] [--split val]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader
import json

from src.config import (
    Config, TUMOR_TYPES, POSITIONS, QUANTITIES,
    OFFSET_TUMOR, OFFSET_POS, OFFSET_QTY, OFFSET_LIVER, LIVER_TYPES,
    PAD, BOS, EOS,
)
from src.model import MRIReportGenerator
from src.dataset import MRISeqDataset, collate_fn
from src.engine import parse_sequence

# ── helpers ───────────────────────────────────────────────────────────────────

def tok2type(t):
    if OFFSET_TUMOR <= t < OFFSET_POS:  return TUMOR_TYPES[t - OFFSET_TUMOR]
    return f"[{t}]"

def tok2pos(p):
    if OFFSET_POS <= p < OFFSET_QTY:   return POSITIONS[p - OFFSET_POS]
    return f"[{p}]"

def tok2qty(q):
    if OFFSET_QTY <= q < OFFSET_QTY + len(QUANTITIES): return QUANTITIES[q - OFFSET_QTY]
    return f"[{q}]"

def fmt_lesion(t, p, q):
    return f"{tok2type(t)}@{tok2pos(p)}(x{tok2qty(q)})"

def fmt_lesions(lesions):
    if not lesions:
        return "NO_LESION"
    return "  |  ".join(fmt_lesion(t, p, q) for t, p, q in lesions)

def pair_f1(pred_pairs, gt_pairs):
    if not gt_pairs and not pred_pairs: return 1.0
    if not gt_pairs or not pred_pairs:  return 0.0
    tp = len(pred_pairs & gt_pairs)
    if tp == 0: return 0.0
    return 2 * tp / (len(pred_pairs) + len(gt_pairs))

def compare_lesions(pred_lesions, gt_lesions):
    """Return (hit_pairs, missed_pairs, false_pairs) as sets of (type,pos)."""
    pred_pairs = {(t, p) for t, p, _ in pred_lesions}
    gt_pairs   = {(t, p) for t, p, _ in gt_lesions}
    hit    = pred_pairs & gt_pairs
    missed = gt_pairs   - pred_pairs
    false  = pred_pairs - gt_pairs
    return hit, missed, false

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",   default="runs/exp_20260302_141015",
                        help="Experiment directory (contains best_model.pth)")
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--top",   type=int, default=20,
                        help="Show top-N cases by pair_F1")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # ── config: detect model size from saved config.json if available ──────
    cfg_json = os.path.join(args.exp, "config.json")
    cfg = Config()
    if os.path.exists(cfg_json):
        with open(cfg_json) as f:
            saved = json.load(f)
        cfg.model_size = saved.get("model_size", "small")
    else:
        cfg.model_size = "small"
    cfg.__post_init__()
    cfg.batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Exp    : {args.exp}")
    print(f"Model  : d_model={cfg.d_model} ({cfg.model_size})")

    # ── load model ────────────────────────────────────────────────────────────
    ckpt = os.path.join(args.exp, "best_model.pth")
    model = MRIReportGenerator(cfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    import time
    mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(ckpt)))
    print(f"Ckpt   : {ckpt}  (saved {mtime})\n")

    # ── load manifest for case IDs ────────────────────────────────────────────
    manifest_path = os.path.join(cfg.cache_root, cfg.cache_tag, "manifest.json")
    case_ids = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            all_records = json.load(f)
        case_ids = [r["id"] for r in all_records if r.get("split") == args.split]

    # ── dataset & loader ──────────────────────────────────────────────────────
    ds = MRISeqDataset(cfg, split=args.split)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate_fn)

    # ── inference ─────────────────────────────────────────────────────────────
    all_cases = []
    pair_gt_counter   = Counter()
    pair_hit_counter  = Counter()
    pair_pred_counter = Counter()

    with torch.no_grad():
        idx = 0
        for batch_x, batch_y, _ in loader:
            batch_x = batch_x.to(device)
            preds = model.generate(batch_x, max_len=batch_y.shape[1] + 2)
            for i in range(len(preds)):
                pred_info = parse_sequence(preds[i].cpu().numpy())
                gt_info   = parse_sequence(batch_y[i].numpy())

                gt_pairs   = {(t, p) for t, p, _ in gt_info["lesions"]}
                pred_pairs = {(t, p) for t, p, _ in pred_info["lesions"]}
                hit, missed, false = compare_lesions(pred_info["lesions"], gt_info["lesions"])

                for pair in gt_pairs:   pair_gt_counter[pair]  += 1
                for pair in pred_pairs: pair_pred_counter[pair] += 1
                for pair in hit:        pair_hit_counter[pair]  += 1

                case_id = case_ids[idx] if idx < len(case_ids) else f"case_{idx}"
                all_cases.append({
                    "idx":      idx,
                    "id":       case_id,
                    "gt":       gt_info["lesions"],
                    "pred":     pred_info["lesions"],
                    "gt_pairs":   gt_pairs,
                    "pred_pairs": pred_pairs,
                    "hit":      hit,
                    "missed":   missed,
                    "false":    false,
                    "pf1":      pair_f1(pred_pairs, gt_pairs),
                    "lesion1p": len(gt_pairs) > 0,
                })
                idx += 1

    # ── split ─────────────────────────────────────────────────────────────────
    lesion1p = [c for c in all_cases if c["lesion1p"]]
    noles    = [c for c in all_cases if not c["lesion1p"]]

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: Overall summary
    # ══════════════════════════════════════════════════════════════════════════
    W = 70
    print("=" * W)
    print("  OVERALL SUMMARY")
    print("=" * W)
    print(f"  Split          : {args.split}  ({len(all_cases)} cases)")
    print(f"  Lesion1p cases : {len(lesion1p)}")
    print(f"  No-lesion cases: {len(noles)}")
    mean_f1 = np.mean([c["pf1"] for c in lesion1p]) if lesion1p else 0
    print(f"  Mean pair_F1 (lesion1p): {mean_f1:.4f}")
    cases_with_hit = sum(1 for c in lesion1p if c["hit"])
    print(f"  Cases with ≥1 correct pair: {cases_with_hit} / {len(lesion1p)} ({100*cases_with_hit/max(len(lesion1p),1):.1f}%)")
    total_gt  = sum(pair_gt_counter.values())
    total_hit = sum(pair_hit_counter.values())
    print(f"  Pair recall (instance-level): {total_hit}/{total_gt} = {total_hit/max(total_gt,1):.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: Per-pair hit rate
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * W)
    print("  PER-PAIR STATISTICS  (sorted by GT frequency)")
    print("=" * W)
    print(f"  {'Pair':<32} {'GT':>4} {'Hit':>4} {'Pred':>5}  {'Recall':>7}  {'Prec':>6}")
    print(f"  {'-'*32} {'-'*4} {'-'*4} {'-'*5}  {'-'*7}  {'-'*6}")
    for pair, gt_cnt in sorted(pair_gt_counter.items(), key=lambda x: -x[1]):
        t, p = pair
        name = f"{tok2type(t)}@{tok2pos(p)}"
        hit  = pair_hit_counter.get(pair, 0)
        pred = pair_pred_counter.get(pair, 0)
        rec  = hit / gt_cnt if gt_cnt else 0
        prec = hit / pred   if pred   else 0
        flag = " ✓" if hit > 0 else ""
        print(f"  {name:<32} {gt_cnt:>4} {hit:>4} {pred:>5}  {rec:>7.2f}  {prec:>6.2f}{flag}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: Model's favourite predictions (bias check)
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * W)
    print("  MODEL PREDICTION BIAS  (top 12 most-predicted pairs)")
    print("=" * W)
    print(f"  {'Pair':<32} {'Pred':>5} {'GT':>4} {'Hit':>4}  Status")
    print(f"  {'-'*32} {'-'*5} {'-'*4} {'-'*4}  {'------'}")
    for pair, cnt in pair_pred_counter.most_common(12):
        t, p = pair
        name  = f"{tok2type(t)}@{tok2pos(p)}"
        gt_c  = pair_gt_counter.get(pair, 0)
        hit_c = pair_hit_counter.get(pair, 0)
        if hit_c == 0:
            status = "❌ never correct"
        elif cnt > gt_c * 2:
            status = "⚠️  over-predicted"
        else:
            status = "✓"
        print(f"  {name:<32} {cnt:>5} {gt_c:>4} {hit_c:>4}  {status}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: Case-by-case comparison (lesion1p, sorted by F1 desc)
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * W)
    print(f"  CASE-BY-CASE COMPARISON  (lesion1p, top {args.top} by pair_F1)")
    print("=" * W)

    sorted_cases = sorted(lesion1p, key=lambda c: -c["pf1"])
    for c in sorted_cases[:args.top]:
        f1_str  = f"{c['pf1']:.3f}"
        hit_n   = len(c["hit"])
        gt_n    = len(c["gt_pairs"])

        print(f"\n  ┌─ [{c['idx']:3d}] {c['id']}  pair_F1={f1_str}  ({hit_n}/{gt_n} pairs correct)")
        print(f"  │  GT  : {fmt_lesions(c['gt'])}")
        print(f"  │  PRED: {fmt_lesions(c['pred'])}")

        if c["hit"]:
            hits = ", ".join(f"{tok2type(t)}@{tok2pos(p)}" for t,p in sorted(c["hit"], key=lambda x: tok2type(x[0])))
            print(f"  │  ✓ Correct   : {hits}")
        if c["missed"]:
            miss = ", ".join(f"{tok2type(t)}@{tok2pos(p)}" for t,p in sorted(c["missed"], key=lambda x: tok2type(x[0])))
            print(f"  │  ✗ Missed    : {miss}")
        if c["false"]:
            fp = ", ".join(f"{tok2type(t)}@{tok2pos(p)}" for t,p in sorted(c["false"], key=lambda x: tok2type(x[0])))
            print(f"  │  ⚠ False pos : {fp}")
        print(f"  └{'─'*60}")

    # ── cases with F1=0 summary ───────────────────────────────────────────────
    zero_cases = [c for c in lesion1p if c["pf1"] == 0.0]
    print(f"\n  Lesion1p cases with pair_F1=0: {len(zero_cases)} / {len(lesion1p)}")
    if zero_cases:
        print(f"  (Most common GT pairs in failed cases:)")
        failed_gt = Counter()
        for c in zero_cases:
            for t, p in c["gt_pairs"]:
                failed_gt[(t,p)] += 1
        for pair, cnt in failed_gt.most_common(8):
            t, p = pair
            print(f"    {tok2type(t)}@{tok2pos(p)}: {cnt} cases")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: No-lesion confusion
    # ══════════════════════════════════════════════════════════════════════════
    noles_correct = sum(1 for c in noles if not c["pred"])
    les1p_wrongly_clean = sum(1 for c in lesion1p if not c["pred"])
    print()
    print("=" * W)
    print("  DETECTION STATUS")
    print("=" * W)
    print(f"  No-lesion GT  → pred no-lesion : {noles_correct}/{len(noles)} ({100*noles_correct/max(len(noles),1):.1f}%) correct")
    print(f"  No-lesion GT  → pred has lesion: {len(noles)-noles_correct}/{len(noles)} (false alarms)")
    print(f"  Lesion1p GT   → pred no-lesion : {les1p_wrongly_clean}/{len(lesion1p)} (missed entirely)")
    print(f"  Lesion1p GT   → pred has lesion: {len(lesion1p)-les1p_wrongly_clean}/{len(lesion1p)} (at least attempted)")
    print()

if __name__ == "__main__":
    main()
