import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from collections import Counter
from .config import PAD, BOS, EOS, NO_LESION, OFFSET_LIVER, OFFSET_TUMOR, OFFSET_POS, OFFSET_QTY, VOCAB_SIZE, TUMOR_TYPES, POSITIONS, QUANTITIES

# Ordinal bucket ordering for quantity tokens (used in soft triplet & qty MAE)
# Token ID → bucket index [0, 1, 2, 3]  ("1"→0, "2"→1, "GE3"→2, "Multiple"→3)
QTY_BUCKET = {OFFSET_QTY + i: i for i in range(len(QUANTITIES))}

# Minimum GT-positive support to include a type in weighted Sen/Spe aggregate.
# Classes with fewer GT-positive cases than this are excluded from the aggregate
# (still reported individually in evaluate_nlg.py).
MIN_SEN_SUPPORT = 5

def train_one_epoch(model, loader, optimizer, device, epoch, loss_fn=None,
                    ss_prob: float = 0.0, word_dropout_p: float = 0.15,
                    scaler=None, aux_weight: float = 2.0):
    """
    Train one epoch.

    ss_prob: scheduled-sampling probability (0 = full teacher forcing).
      With probability ss_prob, the decoder input for the current batch is
      replaced by the model's own greedy predictions (generated under no_grad),
      while the target still uses ground-truth tokens.  This reduces exposure
      bias without requiring slow token-by-token iteration.
    """
    model.train()
    seq_nll_meter = 0
    aux_loss_meter = 0
    criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss(ignore_index=PAD)

    pbar = tqdm(loader, desc=f"Train Ep {epoch}" + (f" ss={ss_prob:.2f}" if ss_prob > 0 else ""))
    for x, y, _ in pbar:
        x, y = x.to(device), y.to(device)

        # Scheduled sampling: per-sample — each sample independently uses
        # its own generated tokens with probability ss_prob.
        # This avoids batch-level all-or-nothing switching which causes
        # high gradient variance and metric oscillation.
        if ss_prob > 0:
            ss_mask = torch.rand(x.shape[0], device=device) < ss_prob  # (B,)
            if ss_mask.any():
                with torch.no_grad():
                    model.eval()
                    gen = model.generate(x[ss_mask], max_len=y.shape[1])  # (B', L')
                    model.train()
                tgt_len = y.shape[1] - 1
                # Align generated length
                if gen.shape[1] - 1 >= tgt_len:
                    gen_input = gen[:, :tgt_len]
                else:
                    pad = torch.full(
                        (gen.shape[0], tgt_len - (gen.shape[1] - 1)),
                        PAD, dtype=torch.long, device=device
                    )
                    gen_input = torch.cat([gen[:, :-1], pad], dim=1)
                dec_input = y[:, :-1].clone()
                dec_input[ss_mask] = gen_input
            else:
                dec_input = y[:, :-1]
        else:
            # Standard teacher forcing
            dec_input = y[:, :-1]

        target = y[:, 1:]

        # Word Dropout: randomly mask decoder input tokens (except BOS at pos 0).
        # Configurable via word_dropout_p argument (was hardcoded 0.15 before).
        if word_dropout_p > 0:
            mask = (torch.rand_like(dec_input.float()) < word_dropout_p)
            mask[:, 0] = False  # always keep BOS
            mask = mask & (dec_input != PAD)  # don't mask padding
            dec_input = dec_input.masked_fill(mask, PAD)

        optimizer.zero_grad()

        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, aux_logit = model(x, dec_input)
            seq_nll = criterion(logits.reshape(-1, VOCAB_SIZE), target.reshape(-1))
            has_lesion = (target >= OFFSET_TUMOR).any(dim=1).float()
            aux_loss = nn.functional.binary_cross_entropy_with_logits(aux_logit, has_lesion)
            loss = seq_nll + aux_weight * aux_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        seq_nll_meter += seq_nll.item()
        aux_loss_meter += aux_loss.item()
        pbar.set_postfix(seq_nll=seq_nll_meter/(pbar.n+1), aux=aux_loss_meter/(pbar.n+1))

    # Gradient norm diagnostics: compare backbone vs decoder to check visual learning
    backbone_grad = sum(
        p.grad.norm().item() ** 2
        for p in model.backbone.parameters() if p.grad is not None
    ) ** 0.5
    decoder_grad = sum(
        p.grad.norm().item() ** 2
        for p in model.decoder.parameters() if p.grad is not None
    ) ** 0.5
    print(f"  [GradNorm] backbone={backbone_grad:.4f}  decoder={decoder_grad:.4f}  "
          f"ratio={backbone_grad/decoder_grad:.3f}" if decoder_grad > 0 else "")

    return seq_nll_meter / len(loader), aux_loss_meter / len(loader)

def parse_sequence(seq):
    """Parses a list of token IDs into structured info"""
    seq = [t for t in seq if t not in [PAD, BOS, EOS]]
    info = {'liver': None, 'lesions': [], 'no_lesion': False}

    if NO_LESION in seq:
        info['no_lesion'] = True
        seq = [t for t in seq if t != NO_LESION]
    
    # 1. Extract Liver (First valid token usually)
    if seq and OFFSET_LIVER <= seq[0] < OFFSET_TUMOR:
        info['liver'] = seq[0]
        seq = seq[1:] # Consume
        
    # 2. Extract triplets
    # Expect pattern: Tumor, Pos, Qty, Tumor, Pos, Qty...
    i = 0
    while i < len(seq) - 2:
        t, p, q = seq[i], seq[i + 1], seq[i + 2]
        if (OFFSET_TUMOR <= t < OFFSET_POS) and (OFFSET_POS <= p < OFFSET_QTY) and (OFFSET_QTY <= q < VOCAB_SIZE):
            info['lesions'].append((t, p, q))
            i += 3
        else:
            # Malformed sequence, skip one
            i += 1
    return info


def f1_from_sets(pred_set, gt_set):
    if len(pred_set) == 0 and len(gt_set) == 0:
        return 1.0
    tp = len(pred_set & gt_set)
    fp = len(pred_set) - tp
    fn = len(gt_set) - tp
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0


def f1_from_counters(pred_ctr: Counter, gt_ctr: Counter):
    if not pred_ctr and not gt_ctr:
        return 1.0, 0, 0, 0
    tp = 0
    for k in set(pred_ctr.keys()) | set(gt_ctr.keys()):
        tp += min(pred_ctr.get(k, 0), gt_ctr.get(k, 0))
    pred_total = sum(pred_ctr.values())
    gt_total = sum(gt_ctr.values())
    fp = pred_total - tp
    fn = gt_total - tp
    denom = (2 * tp + fp + fn)
    f1 = (2 * tp / denom) if denom > 0 else 0.0
    return f1, tp, fp, fn

def soft_triplet_f1_score(pred_lesions: list, gt_lesions: list,
                          alpha_adj: float = 0.5) -> float:
    """
    Soft triplet F1 with quantity ordinal partial credit.

    For each (pred, gt) pair:
      - exact (type, pos, qty) match              → credit 1.0
      - (type, pos) correct, |qty_bucket| == 1   → credit alpha_adj (default 0.5)
      - (type, pos) correct, |qty_bucket| >= 2   → credit 0.0
      - type or pos mismatch                      → credit 0.0

    Uses greedy optimal assignment (sort matches by credit desc, assign 1:1).
    This rewards partial correctness in the hard quantity sub-task while still
    distinguishing it from full correctness.

    Returns scalar F1 in [0, 1]. Both-empty → 1.0.
    """
    if not pred_lesions and not gt_lesions:
        return 1.0
    if not pred_lesions or not gt_lesions:
        return 0.0

    # Build all candidate (credit, gi, pi) triples
    candidates = []
    for gi, g in enumerate(gt_lesions):
        for pi, p in enumerate(pred_lesions):
            if p[0] == g[0] and p[1] == g[1]:          # type & position match
                qty_dist = abs(QTY_BUCKET.get(p[2], 0) - QTY_BUCKET.get(g[2], 0))
                credit = 1.0 if qty_dist == 0 else (alpha_adj if qty_dist == 1 else 0.0)
                if credit > 0:
                    candidates.append((credit, gi, pi))

    # Greedy 1-to-1 assignment: highest credit first
    candidates.sort(key=lambda x: -x[0])
    used_g, used_p = set(), set()
    soft_tp = 0.0
    for credit, gi, pi in candidates:
        if gi not in used_g and pi not in used_p:
            soft_tp += credit
            used_g.add(gi)
            used_p.add(pi)

    prec = soft_tp / len(pred_lesions)
    rec  = soft_tp / len(gt_lesions)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def evaluate(model, loader, device, common_pairs: set = None):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    
    nll_loss = 0
    
    # Metrics
    liver_acc = []
    type_f1 = []
    pair_f1 = []
    triplet_f1 = []
    soft_triplet_f1_list = []          # partial-credit triplet F1

    n_types = len(TUMOR_TYPES)
    n_pos = len(POSITIONS)
    # Per-type presence: TP/FP/FN/TN for binary "does type T appear in this case?"
    type_tp = np.zeros(n_types, dtype=np.float64)
    type_fp = np.zeros(n_types, dtype=np.float64)
    type_fn = np.zeros(n_types, dtype=np.float64)
    type_tn = np.zeros(n_types, dtype=np.float64)
    # Per-position presence (for macro F1; Sen/Spe computed analogously if needed)
    pos_tp = np.zeros(n_pos, dtype=np.float64)
    pos_fp = np.zeros(n_pos, dtype=np.float64)
    pos_fn = np.zeros(n_pos, dtype=np.float64)
    pos_tn = np.zeros(n_pos, dtype=np.float64)

    # Quantity metrics (all conditioned on correct (type, position) pair)
    qty_match       = 0   # exact qty match
    qty_within1     = 0   # qty within 1 ordinal bucket of GT
    qty_ordinal_dist = 0  # sum of ordinal bucket distances (for MAE)
    pair_match = 0

    no_lesion_correct = 0
    no_lesion_total = 0

    pair_tp_total = 0
    pair_fp_total = 0
    pair_fn_total = 0
    # Case-level sensitivity: among truly lesion-positive cases
    case_tp_sen = 0   # gt has lesion AND pred has lesion
    case_fn_sen = 0   # gt has lesion AND pred has NO lesion (missed detection)
    # Case-level specificity: among truly no-lesion cases
    case_tn = 0  # gt no-lesion AND pred no-lesion
    case_fp_spe = 0  # gt no-lesion AND pred has lesion (false alarm)

    trip_tp_total = 0
    trip_fp_total = 0
    trip_fn_total = 0

    pair_f1_zero = []
    pair_f1_nonzero = []
    pair_f1_nonzero_common = []   # filtered: only common pairs counted
    triplet_f1_zero = []
    triplet_f1_nonzero = []

    pair_f1_diffuse = []
    pair_f1_focal = []
    triplet_f1_diffuse = []
    triplet_f1_focal = []

    diffuse_set = {"L_DIFFUSE", "R_DIFFUSE", "DIFFUSE"}
    
    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="Eval"):
            x, y = x.to(device), y.to(device)
            
            # 1. Calculate NLL Loss (Teacher Forcing; skipped for classifier baseline)
            try:
                dec_input = y[:, :-1]
                target = y[:, 1:]
                logits, _ = model(x, dec_input)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), target.reshape(-1))
                nll_loss += loss.item()
            except TypeError:
                pass  # IndependentClassifierBaseline does not support teacher-forcing NLL
            
            # 2. Greedy Decoding for Metrics
            preds = model.generate(x, max_len=y.shape[1] + 2)
            
            # 3. Compute Accuracy/F1
            preds_np = preds.cpu().numpy()
            y_np = y.cpu().numpy()
            
            for i in range(len(preds_np)):
                pred_info = parse_sequence(preds_np[i])
                gt_info = parse_sequence(y_np[i])
                
                # Liver Acc
                if gt_info['liver'] is not None:
                    match = (pred_info['liver'] == gt_info['liver'])
                    liver_acc.append(1.0 if match else 0.0)
                
                # No-lesion accuracy: does model correctly predict "no lesion"?
                # Use lesion count (not NO_LESION token) since that token never appears in GT.
                gt_no   = (len(gt_info["lesions"]) == 0)
                pred_no = (len(pred_info["lesions"]) == 0)
                no_lesion_total += 1
                if gt_no == pred_no:
                    no_lesion_correct += 1

                # Type-F1 (presence of tumor type, ignore position/qty)
                pred_types = {t for (t, _, _) in pred_info["lesions"]}
                gt_types = {t for (t, _, _) in gt_info["lesions"]}
                type_f1.append(f1_from_sets(pred_types, gt_types))

                pred_types_idx = {t - OFFSET_TUMOR for (t, _, _) in pred_info["lesions"]}
                gt_types_idx = {t - OFFSET_TUMOR for (t, _, _) in gt_info["lesions"]}
                for ti in range(n_types):
                    in_pred = ti in pred_types_idx
                    in_gt   = ti in gt_types_idx
                    if     in_pred and     in_gt: type_tp[ti] += 1
                    elif   in_pred and not in_gt: type_fp[ti] += 1
                    elif not in_pred and   in_gt: type_fn[ti] += 1
                    else:                         type_tn[ti] += 1  # both absent

                pred_pos_idx = {p - OFFSET_POS for (_, p, _) in pred_info["lesions"]}
                gt_pos_idx = {p - OFFSET_POS for (_, p, _) in gt_info["lesions"]}
                for pi in range(n_pos):
                    in_pred = pi in pred_pos_idx
                    in_gt   = pi in gt_pos_idx
                    if     in_pred and     in_gt: pos_tp[pi] += 1
                    elif   in_pred and not in_gt: pos_fp[pi] += 1
                    elif not in_pred and   in_gt: pos_fn[pi] += 1
                    else:                         pos_tn[pi] += 1

                # Pair-F1 on (type, position) using set presence
                pred_pairs = {(t, p) for (t, p, _) in pred_info["lesions"]}
                gt_pairs = {(t, p) for (t, p, _) in gt_info["lesions"]}
                pair_f1_current = f1_from_sets(pred_pairs, gt_pairs)
                pair_f1.append(pair_f1_current)

                pair_tp = len(pred_pairs & gt_pairs)
                pair_fp = len(pred_pairs - gt_pairs)
                pair_fn = len(gt_pairs - pred_pairs)
                pair_tp_total += pair_tp
                pair_fp_total += pair_fp
                pair_fn_total += pair_fn

                # Case-level sensitivity (lesion-positive GT cases)
                # Case-level specificity (no-lesion GT cases)
                gt_has_lesion   = len(gt_info["lesions"]) > 0
                pred_has_lesion = len(pred_info["lesions"]) > 0
                if gt_has_lesion:
                    if pred_has_lesion:
                        case_tp_sen += 1   # correctly detected lesion
                    else:
                        case_fn_sen += 1   # missed: GT has lesion, pred says none
                else:
                    if not pred_has_lesion:
                        case_tn += 1       # correctly ruled out lesion
                    else:
                        case_fp_spe += 1   # false alarm: GT clean, pred fires

                # Triplet-F1 on (type, position, quantity) with multiplicity
                pred_trip = Counter(pred_info["lesions"])
                gt_trip = Counter(gt_info["lesions"])
                tf1, ttp, tfp, tfn = f1_from_counters(pred_trip, gt_trip)
                triplet_f1.append(tf1)
                trip_tp_total += ttp
                trip_fp_total += tfp
                trip_fn_total += tfn

                # Soft triplet F1: partial credit for quantity near-misses
                stf1 = soft_triplet_f1_score(pred_info["lesions"], gt_info["lesions"])
                soft_triplet_f1_list.append(stf1)

                # Qty metrics conditioned on correct (type, position)
                # Build qty maps: (type, pos) → set of qty tokens
                pred_qty_map = {}
                for t, p, q in pred_info["lesions"]:
                    pred_qty_map.setdefault((t, p), set()).add(q)
                gt_qty_map = {}
                for t, p, q in gt_info["lesions"]:
                    gt_qty_map.setdefault((t, p), set()).add(q)

                pair_tp_keys = pred_pairs & gt_pairs
                pair_match += len(pair_tp_keys)
                for key in pair_tp_keys:
                    pred_qtys = pred_qty_map.get(key, set())
                    gt_qtys   = gt_qty_map.get(key, set())
                    # Exact qty match
                    if pred_qtys & gt_qtys:
                        qty_match += 1
                    # Ordinal distance: compare best (closest) pred qty to any gt qty
                    if pred_qtys and gt_qtys:
                        min_dist = min(
                            abs(QTY_BUCKET.get(pq, 0) - QTY_BUCKET.get(gq, 0))
                            for pq in pred_qtys for gq in gt_qtys
                        )
                        qty_ordinal_dist += min_dist
                        if min_dist <= 1:
                            qty_within1 += 1

                # Subset metrics
                gt_lesion_count = len(gt_info["lesions"])
                if gt_lesion_count == 0:
                    pair_f1_zero.append(pair_f1_current)
                    triplet_f1_zero.append(tf1)
                else:
                    pair_f1_nonzero.append(pair_f1_current)
                    triplet_f1_nonzero.append(tf1)
                    # Filtered metric: mask out rare pairs from both pred and gt
                    if common_pairs is not None:
                        cp_pred = {(t, p) for t, p in pred_pairs if (t, p) in common_pairs}
                        cp_gt   = {(t, p) for t, p in gt_pairs   if (t, p) in common_pairs}
                        if cp_gt:  # only include case if it has at least one common GT pair
                            pair_f1_nonzero_common.append(f1_from_sets(cp_pred, cp_gt))

                    gt_pos_names = [POSITIONS[p - OFFSET_POS] for (_, p, _) in gt_info["lesions"]]
                    is_diffuse_only = all(name in diffuse_set for name in gt_pos_names)
                    if is_diffuse_only:
                        pair_f1_diffuse.append(pair_f1_current)
                        triplet_f1_diffuse.append(tf1)
                    else:
                        pair_f1_focal.append(pair_f1_current)
                        triplet_f1_focal.append(tf1)

    # ── Macro F1 by type/position (presence-based) ────────────────────────────
    type_macro_f1 = []
    for i in range(n_types):
        denom = 2 * type_tp[i] + type_fp[i] + type_fn[i]
        if denom > 0:
            type_macro_f1.append(2 * type_tp[i] / denom)
    pos_macro_f1 = []
    for i in range(n_pos):
        denom = 2 * pos_tp[i] + pos_fp[i] + pos_fn[i]
        if denom > 0:
            pos_macro_f1.append(2 * pos_tp[i] / denom)

    # ── Per-type Sensitivity & Specificity ───────────────────────────────────
    # Sen_T = TP_T / (TP_T + FN_T)   "when T is present, how often detected?"
    # Spe_T = TN_T / (TN_T + FP_T)   "when T is absent, how often not fired?"
    #
    # Aggregate = PREVALENCE-WEIGHTED average, not macro.
    # Rationale: with HEM having only ~8 test cases, an unweighted macro average
    # is dominated by noise. Weighting by GT-positive support gives results that
    # reflect clinical impact (common types matter more).
    # Types with fewer than MIN_SEN_SUPPORT GT-positive cases are excluded from
    # the aggregate (they are still available per-type in evaluate_nlg.py).
    per_type_sen = {}
    per_type_spe = {}
    type_sen_num = 0.0;  type_sen_den = 0.0
    type_spe_num = 0.0;  type_spe_den = 0.0

    for i, tname in enumerate(TUMOR_TYPES):
        support = type_tp[i] + type_fn[i]   # GT-positive cases for type T
        if support > 0:
            sen_i = float(type_tp[i] / support)
            per_type_sen[tname] = sen_i
            if support >= MIN_SEN_SUPPORT:          # enough support → include
                type_sen_num += sen_i * support
                type_sen_den += support

        neg_support = type_tn[i] + type_fp[i]       # GT-negative cases for type T
        if neg_support > 0:
            spe_i = float(type_tn[i] / neg_support)
            per_type_spe[tname] = spe_i
            if support >= MIN_SEN_SUPPORT:           # gate on same support as Sen
                type_spe_num += spe_i * neg_support
                type_spe_den += neg_support

    type_sen_weighted = type_sen_num / type_sen_den if type_sen_den > 0 else 0.0
    type_spe_weighted = type_spe_num / type_spe_den if type_spe_den > 0 else 0.0

    # ── Case-level Sensitivity & Specificity ─────────────────────────────────
    case_sen = case_tp_sen / (case_tp_sen + case_fn_sen) if (case_tp_sen + case_fn_sen) > 0 else 0.0
    case_spe = case_tn   / (case_tn   + case_fp_spe)    if (case_tn   + case_fp_spe)  > 0 else 0.0

    pair_f1_micro = (2 * pair_tp_total / (2 * pair_tp_total + pair_fp_total + pair_fn_total)) if (pair_tp_total + pair_fp_total + pair_fn_total) > 0 else 0.0
    triplet_f1_micro = (2 * trip_tp_total / (2 * trip_tp_total + trip_fp_total + trip_fn_total)) if (trip_tp_total + trip_fp_total + trip_fn_total) > 0 else 0.0

    metrics = {
        "nll": nll_loss / len(loader),
        "liver_acc":    np.mean(liver_acc)    if liver_acc    else 0.0,
        "no_lesion_acc": (no_lesion_correct / no_lesion_total) if no_lesion_total > 0 else 0.0,
        "type_f1":      np.mean(type_f1)      if type_f1      else 0.0,

        # ── Pair / Triplet F1 ──────────────────────────────────────────────
        "pair_f1":         np.mean(pair_f1)    if pair_f1    else 0.0,
        "pair_f1_micro":   pair_f1_micro,
        # Strict triplet (exact type + pos + qty)
        "triplet_f1":      np.mean(triplet_f1) if triplet_f1 else 0.0,
        "triplet_f1_micro": triplet_f1_micro,
        # Soft triplet: partial credit when qty is within 1 ordinal bucket
        # (rewards "1→2" less than exact, but more than "1→Multiple")
        "soft_triplet_f1": np.mean(soft_triplet_f1_list) if soft_triplet_f1_list else 0.0,

        # ── Case-level Sen / Spe ───────────────────────────────────────────
        "case_sen": case_sen,   # recall: GT has lesion → model detects ≥1
        "case_spe": case_spe,   # specificity: GT clean → model stays quiet

        # ── Per-type Sen / Spe (prevalence-weighted aggregate) ────────────
        # Weighted by GT-positive support; types with <MIN_SEN_SUPPORT excluded.
        # This avoids HEM (8 cases) distorting the aggregate.
        "type_sen_weighted": type_sen_weighted,
        "type_spe_weighted": type_spe_weighted,
        # Full per-type breakdown (dict, not written to CSV log)
        "per_type_sen": per_type_sen,
        "per_type_spe": per_type_spe,

        # ── Quantity metrics (conditioned on correct pair) ─────────────────
        # Exact match
        "qty_acc":         (qty_match   / pair_match) if pair_match > 0 else 0.0,
        # Within-1-bucket match (ordinal tolerance: "1↔2" accepted, "1↔Multiple" not)
        "qty_within1_acc": (qty_within1 / pair_match) if pair_match > 0 else 0.0,
        # Mean Absolute Error in bucket space [0,1,2,3]
        "qty_ordinal_mae": (qty_ordinal_dist / pair_match) if pair_match > 0 else 0.0,

        # ── Auxiliary ─────────────────────────────────────────────────────
        "type_macro_f1": float(np.mean(type_macro_f1)) if type_macro_f1 else 0.0,
        "pos_macro_f1":  float(np.mean(pos_macro_f1))  if pos_macro_f1  else 0.0,

        # ── Subset breakdowns ──────────────────────────────────────────────
        "pair_f1_lesion0":          float(np.mean(pair_f1_zero))           if pair_f1_zero           else 0.0,
        "pair_f1_lesion1p":         float(np.mean(pair_f1_nonzero))        if pair_f1_nonzero        else 0.0,
        "pair_f1_lesion1p_common":  float(np.mean(pair_f1_nonzero_common)) if pair_f1_nonzero_common else 0.0,
        "pair_f1_diffuse":     float(np.mean(pair_f1_diffuse))  if pair_f1_diffuse else 0.0,
        "pair_f1_focal":       float(np.mean(pair_f1_focal))    if pair_f1_focal   else 0.0,
        "triplet_f1_lesion0":  float(np.mean(triplet_f1_zero))  if triplet_f1_zero    else 0.0,
        "triplet_f1_lesion1p": float(np.mean(triplet_f1_nonzero)) if triplet_f1_nonzero else 0.0,
        "triplet_f1_diffuse":  float(np.mean(triplet_f1_diffuse)) if triplet_f1_diffuse else 0.0,
        "triplet_f1_focal":    float(np.mean(triplet_f1_focal))   if triplet_f1_focal   else 0.0,
    }
    return metrics
