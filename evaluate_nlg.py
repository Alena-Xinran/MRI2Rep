"""
NLG Evaluation for MRI2Rep
===========================
Converts model predictions (structured token sequences) to radiology-style
natural language reports, then computes NLG metrics vs. real clinical reports.

Pipeline (RadGPT-style naturalization + CT2Rep-style metrics):

  Step 1 — Model inference on the test split (greedy decoding).
  Step 2 — seq_to_report(): template-based converter turns each token sequence
            into a clinical paragraph, closely following the Yale liver MRI
            report writing style learned from liver_report_all.csv.
  Step 3 — [Optional] LLM stylization: pass the template output through a local
            LLM to make the language more natural. Use --llm_model to enable.
            See STYLIZE_PROMPT at the bottom of this file for the exact prompt.
  Step 4 — Compute BLEU-1/2/3/4, ROUGE-L, METEOR in two modes:
            Mode A: pred_template_nl vs gt_template_nl (apple-to-apple, label accuracy)
            Mode B: pred_template_nl vs real_report    (CT2Rep-style, literature comparability)
            These NLG metrics are the same class as CT2Rep's evaluation suite.
  Step 5 — Print a per-type Sen/Spe breakdown (richer than the training log).

Usage:
  python evaluate_nlg.py --run_dir runs/exp_YYYYMMDD_HHMMSS
  python evaluate_nlg.py --run_dir runs/... --ckpt best_model.pth
  python evaluate_nlg.py --run_dir runs/... --llm_model <local-model-name>

Outputs:
  <run_dir>/nlg_eval_results.json   — all scores (both NLG modes + Sen/Spe)
  <run_dir>/nlg_eval_reports.csv    — per-case: GT seq, pred seq, GT template NL,
                                      pred template NL, pred stylized NL, real report
"""

import os
import sys
import json
import csv
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# NLG metric libraries
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as _meteor_score
from rouge_score import rouge_scorer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)

from src.config import (
    Config, PAD, BOS, EOS, NO_LESION,
    TUMOR_TYPES, POSITIONS, LIVER_TYPES, QUANTITIES,
    OFFSET_LIVER, OFFSET_TUMOR, OFFSET_POS, OFFSET_QTY,
)
from src.dataset import MRISeqDataset, collate_fn
from src.model import MRIReportGenerator
from src.engine import parse_sequence
from src.utils import set_seed


# ══════════════════════════════════════════════════════════════════════════════
#  Vocabulary-to-clinical-language mappings
#  (learned from Yale liver_report_all.csv style analysis)
# ══════════════════════════════════════════════════════════════════════════════

LIVER_NL = {
    "Fibrosis/Cirrhosis":   "The liver demonstrates morphologic features of cirrhosis.",
    "NoFibrosis/Cirrhosis": "The liver parenchyma demonstrates normal morphology without cirrhotic features.",
}

# Predicate phrase (after "is/are identified in <location>,")
TYPE_PRED = {
    "APHE_WO": (
        "demonstrating arterial phase hyperenhancement with washout on portal-venous phase"
        ", consistent with hepatocellular carcinoma (LR-5)"
    ),
    "RIM_ATYP": (
        "demonstrating rim enhancement with targetoid diffusion restriction"
        ", suspicious for intrahepatic cholangiocarcinoma or metastasis (LR-M)"
    ),
    "APHE_NoWO": (
        "demonstrating arterial phase hyperenhancement without definite washout"
        " or pseudocapsule (LR-3, indeterminate)"
    ),
    "HEM": (
        "demonstrating T2 hyperintensity with peripheral nodular enhancement on delayed phase"
        ", consistent with hemangioma"
    ),
    "CYST": (
        "without internal enhancement or septations, consistent with a simple cyst"
    ),
}

# (singular, plural) noun for each type
TYPE_NOUN = {
    "APHE_WO":   ("lesion",        "lesions"),
    "RIM_ATYP":  ("lesion",        "lesions"),
    "APHE_NoWO": ("focus",         "foci"),
    "HEM":       ("lesion",        "lesions"),
    "CYST":      ("cystic lesion", "cystic lesions"),
}

# Anatomical position descriptions (Couinaud mapping used in real reports)
POS_NL = {
    "L_LAT":      "segment 2/3 (left lateral lobe)",
    "L_MED":      "segment 4 (left medial lobe)",
    "R_ANT":      "segment 5/8 (right anterior sector)",
    "R_POST":     "segment 6/7 (right posterior sector)",
    "R_JUNCTION": "segment 5/6 junction",
    "CAUDATE":    "segment 1 (caudate lobe)",
    "L_DIFFUSE":  "throughout the left lobe",
    "R_DIFFUSE":  "throughout the right lobe",
    "DIFFUSE":    "diffusely throughout the liver",
}

# Quantity → (determiner/prefix, plural?)
QTY_NL = {
    "1":        ("A solitary",  False),
    "2":        ("Two",         True),
    "GE3":      ("At least three", True),
    "Multiple": ("Multiple",    True),
}

DIFFUSE_POSITIONS = {"L_DIFFUSE", "R_DIFFUSE", "DIFFUSE"}


# ══════════════════════════════════════════════════════════════════════════════
#  seq_to_report():  structured token sequence → radiology-style paragraph
# ══════════════════════════════════════════════════════════════════════════════

def seq_to_report(token_ids: list[int]) -> str:
    """
    Convert a list of integer token IDs to a radiology-style clinical paragraph.

    The output mimics the Yale liver MRI report structure:
      1. Liver morphology sentence.
      2. Lesion findings (one sentence per unique lesion type × position).
      3. "No evidence of HCC" if no lesion with washout found.
      4. Hepatic vasculature sentence.
    """
    # Parse using engine helper
    info = parse_sequence(token_ids)
    sentences = []

    # 1. Liver morphology
    if info["liver"] is not None:
        liver_idx = info["liver"] - OFFSET_LIVER
        if 0 <= liver_idx < len(LIVER_TYPES):
            liver_key = LIVER_TYPES[liver_idx]
            sentences.append(LIVER_NL.get(liver_key, ""))

    # 2. Lesion findings
    if not info["lesions"]:
        sentences.append(
            "No arterial phase hyperenhancing lesions with washout are identified to suggest hepatocellular carcinoma."
        )
    else:
        has_aphe_wo = any(
            TUMOR_TYPES[t - OFFSET_TUMOR] == "APHE_WO"
            for t, p, q in info["lesions"]
        )

        for t_tok, p_tok, q_tok in info["lesions"]:
            t_idx = t_tok - OFFSET_TUMOR
            p_idx = p_tok - OFFSET_POS
            q_idx = q_tok - OFFSET_QTY

            if not (0 <= t_idx < len(TUMOR_TYPES)): continue
            if not (0 <= p_idx < len(POSITIONS)):    continue
            if not (0 <= q_idx < len(QUANTITIES)):   continue

            tname = TUMOR_TYPES[t_idx]
            pname = POSITIONS[p_idx]
            qname = QUANTITIES[q_idx]

            det, plural = QTY_NL[qname]
            verb = "are" if plural else "is"
            noun = TYPE_NOUN[tname][1] if plural else TYPE_NOUN[tname][0]
            pred = TYPE_PRED[tname]
            loc  = POS_NL.get(pname, pname)

            if pname in DIFFUSE_POSITIONS:
                sent = f"{det} {noun} {verb} identified {loc}, {pred}."
            else:
                sent = f"{det} {noun} {verb} identified in {loc}, {pred}."

            sentences.append(sent[0].upper() + sent[1:])

        if not has_aphe_wo:
            sentences.append(
                "No additional arterial phase hyperenhancing lesions with washout are identified."
            )

    # 3. Vasculature boilerplate (present in nearly every real report)
    sentences.append("The hepatic vasculature is patent.")

    return " ".join(s for s in sentences if s)


# ══════════════════════════════════════════════════════════════════════════════
#  NLG metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer (consistent with NLG eval norms)."""
    import re
    return re.findall(r"\b\w+\b", text.lower())


def compute_nlg_metrics(
    hypothesis_list: list[str],
    reference_list: list[str],
) -> dict:
    """
    Compute corpus-level BLEU-1/2/3/4, ROUGE-L, METEOR, and BERTScore-F1.

    BLEU/ROUGE/METEOR are n-gram-based and sensitive to template writing style,
    which makes Mode B (pred vs real report) scores artificially low even when
    the predicted clinical content is correct.

    BERTScore uses contextual BERT embeddings and is more robust to style
    differences — it better captures semantic equivalence between, e.g.,
    "demonstrating arterial phase hyperenhancement" and "shows avid arterial
    enhancement", which BLEU would score as near-zero.

    Falls back gracefully if the bert_score library is not installed.
    """
    assert len(hypothesis_list) == len(reference_list)

    # BLEU (corpus-level, smoothed)
    hyp_toks = [tokenize(h) for h in hypothesis_list]
    ref_toks  = [[tokenize(r)] for r in reference_list]
    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(ref_toks, hyp_toks, weights=(1, 0, 0, 0),             smoothing_function=smooth)
    bleu2 = corpus_bleu(ref_toks, hyp_toks, weights=(0.5, 0.5, 0, 0),         smoothing_function=smooth)
    bleu3 = corpus_bleu(ref_toks, hyp_toks, weights=(1/3, 1/3, 1/3, 0),       smoothing_function=smooth)
    bleu4 = corpus_bleu(ref_toks, hyp_toks, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    # ROUGE-L (average over corpus)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL_scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for hyp, ref in zip(hypothesis_list, reference_list)
    ]
    rougeL = float(np.mean(rougeL_scores))

    # METEOR (average over corpus)
    meteor_scores = [
        _meteor_score([tokenize(ref)], tokenize(hyp))
        for hyp, ref in zip(hypothesis_list, reference_list)
    ]
    meteor = float(np.mean(meteor_scores))

    # BERTScore — semantic similarity via contextual embeddings.
    # More robust to paraphrasing / style variation than n-gram metrics.
    # Uses Bio_ClinicalBERT when available (ideal for radiology text), otherwise
    # falls back to bert-base-uncased, and silently skips if not installed.
    bert_f1_score = None
    try:
        from bert_score import score as _bert_score
        # Try clinical BERT first; fall back to base model if download fails
        for bert_model in ("emilyalsentzer/Bio_ClinicalBERT", "bert-base-uncased"):
            try:
                _, _, F = _bert_score(
                    hypothesis_list, reference_list,
                    model_type=bert_model, lang="en", verbose=False,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                bert_f1_score = round(float(F.mean().item()), 4)
                break
            except Exception:
                continue
    except ImportError:
        pass   # bert_score not installed — skip silently

    result = {
        "bleu1":  round(bleu1,  4),
        "bleu2":  round(bleu2,  4),
        "bleu3":  round(bleu3,  4),
        "bleu4":  round(bleu4,  4),
        "rougeL": round(rougeL, 4),
        "meteor": round(meteor, 4),
    }
    if bert_f1_score is not None:
        result["bert_f1"] = bert_f1_score
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Per-type Sen/Spe detailed breakdown (printed, not used as loss)
# ══════════════════════════════════════════════════════════════════════════════

def compute_detailed_sen_spe(pred_infos: list, gt_infos: list) -> dict:
    """
    Returns per-type and per-position sensitivity/specificity tables,
    plus case-level sen/spe. Mirrors the engine.py metrics but gives
    the full per-class breakdown for the final test-set report.
    """
    n_types = len(TUMOR_TYPES)
    n_pos   = len(POSITIONS)
    type_tp = np.zeros(n_types); type_fp = np.zeros(n_types)
    type_fn = np.zeros(n_types); type_tn = np.zeros(n_types)
    pos_tp  = np.zeros(n_pos);   pos_fp  = np.zeros(n_pos)
    pos_fn  = np.zeros(n_pos);   pos_tn  = np.zeros(n_pos)
    case_tp = case_fn = case_tn_s = case_fp_s = 0

    for pred, gt in zip(pred_infos, gt_infos):
        pred_types = {t - OFFSET_TUMOR for t, p, q in pred["lesions"]}
        gt_types   = {t - OFFSET_TUMOR for t, p, q in gt["lesions"]}
        pred_pos   = {p - OFFSET_POS   for t, p, q in pred["lesions"]}
        gt_pos     = {p - OFFSET_POS   for t, p, q in gt["lesions"]}

        for i in range(n_types):
            ip, ig = i in pred_types, i in gt_types
            if   ip and ig:  type_tp[i] += 1
            elif ip:         type_fp[i] += 1
            elif ig:         type_fn[i] += 1
            else:            type_tn[i] += 1

        for i in range(n_pos):
            ip, ig = i in pred_pos, i in gt_pos
            if   ip and ig:  pos_tp[i] += 1
            elif ip:         pos_fp[i] += 1
            elif ig:         pos_fn[i] += 1
            else:            pos_tn[i] += 1

        gt_has  = len(gt["lesions"]) > 0
        pr_has  = len(pred["lesions"]) > 0
        if gt_has:
            if pr_has: case_tp += 1
            else:      case_fn += 1
        else:
            if not pr_has: case_tn_s += 1
            else:          case_fp_s += 1

    def safe_div(a, b): return float(a / b) if b > 0 else float("nan")

    per_type = {}
    for i, name in enumerate(TUMOR_TYPES):
        per_type[name] = {
            "sen": safe_div(type_tp[i], type_tp[i] + type_fn[i]),
            "spe": safe_div(type_tn[i], type_tn[i] + type_fp[i]),
            "TP": int(type_tp[i]), "FP": int(type_fp[i]),
            "FN": int(type_fn[i]), "TN": int(type_tn[i]),
        }

    per_pos = {}
    for i, name in enumerate(POSITIONS):
        per_pos[name] = {
            "sen": safe_div(pos_tp[i], pos_tp[i] + pos_fn[i]),
            "spe": safe_div(pos_tn[i], pos_tn[i] + pos_fp[i]),
            "TP": int(pos_tp[i]), "FP": int(pos_fp[i]),
            "FN": int(pos_fn[i]), "TN": int(pos_tn[i]),
        }

    return {
        "case_sen": safe_div(case_tp, case_tp + case_fn),
        "case_spe": safe_div(case_tn_s, case_tn_s + case_fp_s),
        "per_type": per_type,
        "per_pos":  per_pos,
    }


def print_sen_spe_table(ss: dict):
    print("\n── Case-level Sen/Spe ────────────────────────────────────────")
    print(f"  case_sen = {ss['case_sen']:.3f}   case_spe = {ss['case_spe']:.3f}")

    print("\n── Per Tumor-Type Sen / Spe ──────────────────────────────────")
    print(f"  {'Type':<14}  {'Sen':>6}  {'Spe':>6}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'TN':>5}")
    for name, v in ss["per_type"].items():
        sen = f"{v['sen']:.3f}" if not (isinstance(v['sen'], float) and v['sen'] != v['sen']) else "  n/a"
        spe = f"{v['spe']:.3f}" if not (isinstance(v['spe'], float) and v['spe'] != v['spe']) else "  n/a"
        print(f"  {name:<14}  {sen:>6}  {spe:>6}  {v['TP']:>5}  {v['FP']:>5}  {v['FN']:>5}  {v['TN']:>5}")

    print("\n── Per Position Sen / Spe ────────────────────────────────────")
    print(f"  {'Position':<14}  {'Sen':>6}  {'Spe':>6}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'TN':>5}")
    for name, v in ss["per_pos"].items():
        sen = f"{v['sen']:.3f}" if not (isinstance(v['sen'], float) and v['sen'] != v['sen']) else "  n/a"
        spe = f"{v['spe']:.3f}" if not (isinstance(v['spe'], float) and v['spe'] != v['spe']) else "  n/a"
        print(f"  {name:<14}  {sen:>6}  {spe:>6}  {v['TP']:>5}  {v['FP']:>5}  {v['FN']:>5}  {v['TN']:>5}")


# ══════════════════════════════════════════════════════════════════════════════
#  Optional LLM stylization (local model via subprocess or API)
# ══════════════════════════════════════════════════════════════════════════════

STYLIZE_PROMPT = """\
You are an experienced abdominal radiologist at a major academic medical center.
Below is a structured clinical summary of a liver MRI (gadolinium-enhanced, 3-phase).
Rewrite it as a single, concise paragraph in the style of a formal radiology report.
Follow these rules:
  - Use passive voice and present tense ("is identified", "demonstrates").
  - Do NOT invent findings not present in the summary.
  - Do NOT add clinical recommendations or impressions.
  - Keep it to 3–5 sentences.
  - Do NOT include a heading or bullet points.

Structured summary:
{structured_text}

Radiology paragraph:"""


def llm_stylize(texts: list[str], model_name: str) -> list[str]:
    """
    Optional: pass template outputs through a local LLM for more natural language.
    Requires the 'anthropic' or 'openai' Python SDK, or a local model via subprocess.
    Currently implements Anthropic Claude API as an example; adapt as needed.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()
        out = []
        for text in tqdm(texts, desc=f"LLM stylize ({model_name})"):
            prompt = STYLIZE_PROMPT.format(structured_text=text)
            msg = client.messages.create(
                model=model_name,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            out.append(msg.content[0].text.strip())
        return out
    except Exception as e:
        print(f"[Warning] LLM stylization failed ({e}); using template output as-is.")
        return texts


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NLG evaluation for MRI2Rep")
    parser.add_argument("--run_dir", required=True,
                        help="Path to training run directory")
    parser.add_argument("--ckpt", default="best_model.pth",
                        help="Checkpoint filename inside run_dir (default: best_model.pth)")
    parser.add_argument("--report_csv",
                        default=None,
                        help="CSV with real radiology reports (columns: id, report). "
                             "If omitted, Mode B (pred vs real report) is skipped.")
    parser.add_argument("--split", default="test",
                        choices=["test", "val"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--llm_model", default=None,
                        help="If set, apply LLM stylization using this model name "
                             "(e.g. claude-haiku-4-5-20251001 or a local model ID). "
                             "Metrics are computed on the stylized output.")
    args = parser.parse_args()

    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load real reports ──────────────────────────────────────────────────────
    real_reports = {}
    if args.report_csv is not None:
        with open(args.report_csv, newline="") as f:
            for row in csv.DictReader(f):
                real_reports[row["id"]] = row["report"].strip()
        print(f"Loaded {len(real_reports)} real reports from {args.report_csv}")
    else:
        print("[Info] No --report_csv provided; Mode B (pred vs real report) will be skipped.")

    # ── Dataset & loader ───────────────────────────────────────────────────────
    ds = MRISeqDataset(cfg, split=args.split)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate_fn,
                        pin_memory=True,
                        persistent_workers=(cfg.num_workers > 0),
                        prefetch_factor=2 if cfg.num_workers > 0 else None)
    print(f"Evaluating on {len(ds)} {args.split} samples")

    # ── Load model ─────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.run_dir, args.ckpt)
    if not os.path.exists(ckpt_path):
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    weights = torch.load(ckpt_path, map_location=device)
    # Handle full checkpoint dict vs plain state dict
    if "model" in weights:
        weights = weights["model"]
    # Auto-detect enc_layers from checkpoint
    if "vis_pos_emb" not in weights and cfg.enc_layers > 0:
        cfg.enc_layers = 0
    model = MRIReportGenerator(cfg).to(device)
    model.load_state_dict(weights, strict=True)
    model.eval()
    print(f"Loaded {args.ckpt}")

    # ── Inference ──────────────────────────────────────────────────────────────
    all_pred_seqs   = []   # list of int lists
    all_gt_seqs     = []
    all_rids        = []

    with torch.no_grad():
        for x, y, rids in tqdm(loader, desc="Inference"):
            x = x.to(device)
            preds = model.generate(x, max_len=y.shape[1] + 2)
            for i in range(len(rids)):
                all_pred_seqs.append(preds[i].cpu().tolist())
                all_gt_seqs.append(y[i].tolist())
                all_rids.append(rids[i])

    print(f"Inference done: {len(all_pred_seqs)} predictions")

    # ── Parse sequences ────────────────────────────────────────────────────────
    pred_infos = [parse_sequence(seq) for seq in all_pred_seqs]
    gt_infos   = [parse_sequence(seq) for seq in all_gt_seqs]

    # ── Sen/Spe detailed breakdown ─────────────────────────────────────────────
    ss = compute_detailed_sen_spe(pred_infos, gt_infos)
    print_sen_spe_table(ss)

    # ── Template naturalization (pred + GT) ───────────────────────────────────
    print("Converting sequences to natural language...")
    pred_template_nl = [seq_to_report(seq) for seq in all_pred_seqs]
    gt_template_nl   = [seq_to_report(seq) for seq in all_gt_seqs]

    # ── Optional LLM stylization (applied to pred only, for mode B) ───────────
    if args.llm_model:
        pred_stylized_nl = llm_stylize(pred_template_nl, args.llm_model)
    else:
        pred_stylized_nl = pred_template_nl

    # ── Mode A: pred_template vs gt_template (apple-to-apple, label accuracy) ─
    nlg_vs_template = compute_nlg_metrics(pred_template_nl, gt_template_nl)
    print("\n── NLG Metrics — Mode A: pred_template vs GT_template ───────")
    print("   (same template style; measures whether predicted labels are correct)")
    for k, v in nlg_vs_template.items():
        print(f"  {k:>8} = {v:.4f}")

    # ── Mode B: pred vs real report (CT2Rep-style, matched cases only) ────────
    matched_hyp  = []
    matched_ref  = []
    matched_gt   = []
    matched_ids  = []
    skipped      = 0
    rid_to_idx   = {rid: i for i, rid in enumerate(all_rids)}

    for i, rid in enumerate(all_rids):
        ref = real_reports.get(rid)
        if ref is None:
            skipped += 1
            continue
        matched_hyp.append(pred_stylized_nl[i])
        matched_ref.append(ref)
        matched_gt.append(gt_template_nl[i])
        matched_ids.append(rid)

    if skipped > 0:
        print(f"\n[Warning] {skipped} cases had no matching real report — skipped from Mode B.")
    print(f"\n── NLG Metrics — Mode B: pred vs real report ({len(matched_hyp)} matched cases) ─")
    print("   (CT2Rep-style; measures similarity to actual clinical writing)")

    if matched_hyp:
        nlg_vs_real = compute_nlg_metrics(matched_hyp, matched_ref)
        for k, v in nlg_vs_real.items():
            print(f"  {k:>8} = {v:.4f}")
    else:
        nlg_vs_real = {}
        print("  [No matched cases — skipping Mode B]")

    # ── Save results ───────────────────────────────────────────────────────────
    results = {
        "split":                args.split,
        "n_cases_total":        len(all_rids),
        "n_cases_matched":      len(matched_hyp),
        "ckpt":                 args.ckpt,
        "llm_model":            args.llm_model,
        "nlg_vs_gt_template":   nlg_vs_template,
        "nlg_vs_real_report":   nlg_vs_real,
        "case_sen":             ss["case_sen"],
        "case_spe":             ss["case_spe"],
        "per_type":             ss["per_type"],
        "per_pos":              ss["per_pos"],
    }
    out_json = os.path.join(args.run_dir, "nlg_eval_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results → {out_json}")

    # Per-case CSV for inspection
    out_csv = os.path.join(args.run_dir, "nlg_eval_reports.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "gt_seq", "pred_seq",
            "gt_template_nl", "pred_template_nl",
            "pred_stylized_nl", "real_report"
        ])
        for rid, idx in [(r, rid_to_idx[r]) for r in all_rids]:
            stylized = pred_stylized_nl[idx] if args.llm_model else ""
            writer.writerow([
                rid,
                str(all_gt_seqs[idx]),
                str(all_pred_seqs[idx]),
                gt_template_nl[idx],
                pred_template_nl[idx],
                stylized,
                real_reports.get(rid, ""),
            ])
    print(f"Saved per-case report CSV → {out_csv}")


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
#  LLM stylization prompt reference (full version)
#  Paste this into any LLM API to naturalize the template output.
# ══════════════════════════════════════════════════════════════════════════════
#
# STYLIZE_PROMPT (shown above) takes one argument: {structured_text}.
# Example structured_text:
#   "The liver demonstrates morphologic features of cirrhosis. A solitary lesion
#    is identified in segment 5/8 (right anterior sector), demonstrating arterial
#    phase hyperenhancing with washout appearance on portal-venous phase,
#    consistent with hepatocellular carcinoma (LR-5). The hepatic vasculature
#    is patent."
#
# Expected LLM output:
#   "Cirrhotic liver morphology is again demonstrated. On series 12 image 34,
#    a 1.8 cm arterial phase hyperenhancing mass with washout is identified in
#    segment 8, consistent with hepatocellular carcinoma (LR-5). The hepatic
#    vasculature remains patent."
#
# Note: the LLM is a CONVERTER (structured → natural language), NOT a SCORER.
# All metric computation (BLEU/ROUGE/METEOR) happens on the converted output
# vs. the ground-truth real report — the LLM never sees or scores report quality.
# ══════════════════════════════════════════════════════════════════════════════
