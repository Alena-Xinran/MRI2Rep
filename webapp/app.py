"""
MRI2Rep Web App
Upload ART + PV + Organ Mask (.nii.gz) → structured liver report
"""

import os
import sys
import glob
import uuid
import json
import io
import base64
import tempfile
import traceback

import torch
import numpy as np
import nibabel as nib

from flask import Flask, request, jsonify, render_template, send_file
from monai.transforms import ResizeWithPadOrCrop, ScaleIntensityRangePercentiles
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    Config,
    LIVER_TYPES, TUMOR_TYPES, POSITIONS, QUANTITIES,
    OFFSET_LIVER, OFFSET_TUMOR, OFFSET_POS, OFFSET_QTY,
    PAD, BOS, EOS, NO_LESION, VOCAB_SIZE,
)
from src.model import MRIReportGenerator
from src.dataset import MRISeqDataset, collate_fn
from src.engine import parse_sequence
from torch.utils.data import DataLoader
from collections import Counter

# ── Constants ─────────────────────────────────────────────────────────────────
ROI_SIZE = (192, 192, 96)
MAX_GENERATE_LEN = 30
UPLOAD_FOLDER = tempfile.gettempdir()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB per file


# ── Model loading ─────────────────────────────────────────────────────────────
_model = None
_device = None
_model_info = {"path": None, "error": None}

# ── Val set cache ─────────────────────────────────────────────────────────────
_val_cases   = None   # list of case dicts after inference
_val_manifest = []    # [{id, cache_path, sequence_raw, ...}]

# ── LLM report generator (OpenAI-compatible API) ─────────────────────────────
_llm_client = None
_llm_model  = os.environ.get("LLM_MODEL", "claude-3-5-sonnet-20241022")
_llm_info   = {"model": None, "error": None}

def init_llm():
    global _llm_client, _llm_info
    api_key  = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")
    if not api_key:
        _llm_info["error"] = "OPENAI_API_KEY not set"
        print(f"[LLM] {_llm_info['error']}")
        return
    try:
        from openai import OpenAI
        _llm_client = OpenAI(api_key=api_key, base_url=base_url)
        _llm_info["model"] = _llm_model
        print(f"[LLM] Client ready — model={_llm_model}  base={base_url}")
    except Exception as e:
        _llm_info["error"] = str(e)
        print(f"[LLM] Init failed: {e}")


_LLM_SYSTEM = (
    "You are an experienced abdominal radiologist writing structured liver MRI reports. "
    "Write clearly, concisely, and in standard clinical radiology style. "
    "Use formal medical terminology. Do not invent findings beyond what is provided. "
    "Output only the report text — no headings, no preamble, no markdown."
)

def generate_llm_report(seq_raw: str, template_draft: str = "") -> str | None:
    """Call LLM API to generate a natural-language report from structured findings."""
    if _llm_client is None:
        return None
    try:
        user_msg = (
            "An automated liver MRI analysis system produced the following structured findings:\n\n"
            f"  {seq_raw}\n\n"
        )
        if template_draft:
            user_msg += (
                "A rule-based template produced this draft report:\n\n"
                f"  {template_draft}\n\n"
                "Please rewrite it as a polished, fluent clinical radiology report (2–4 sentences). "
                "Keep all findings; improve phrasing and clinical tone."
            )
        else:
            user_msg += (
                "Write a concise clinical radiology report paragraph (2–4 sentences) "
                "describing these findings in natural language."
            )
        resp = _llm_client.chat.completions.create(
            model=_llm_model,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM] generate error: {e}")
        return None

# ── T5 report generator ───────────────────────────────────────────────────────
_t5_model     = None
_t5_tokenizer = None
_t5_info      = {"path": None, "error": None}

# Search order for T5 checkpoint directory
_T5_SEARCH_PATHS = [
    os.path.join(PROJECT_ROOT, "pretrained", "t5_report_gen"),
    os.path.join(os.path.dirname(PROJECT_ROOT), "MRI2Rep_small_v2", "pretrained", "t5_report_gen"),
    os.path.join(os.path.dirname(PROJECT_ROOT), "MRI2Rep_large",    "pretrained", "t5_report_gen"),
]

# ── Upload volume cache (in-memory, keyed by upload_id) ───────────────────────
_upload_vols = {}     # {upload_id: np.ndarray (3, H, W, D)}

_TYPE_DISPLAY = {
    "APHE_WO":   "APHE + Washout",
    "RIM_ATYP":  "Rim / Atypical",
    "APHE_NoWO": "APHE no-Washout",
    "HEM":       "Hemangioma",
    "CYST":      "Cyst",
}
_POS_DISPLAY = {
    "L_LAT":      "Left Lateral",
    "L_MED":      "Left Medial",
    "R_ANT":      "Right Anterior",
    "R_POST":     "Right Posterior",
    "R_JUNCTION": "Right Junction",
    "CAUDATE":    "Caudate",
    "L_DIFFUSE":  "Left Diffuse",
    "R_DIFFUSE":  "Right Diffuse",
    "DIFFUSE":    "Diffuse (bilateral)",
}
_QTY_DISPLAY = {
    "1": "×1", "2": "×2", "GE3": "×≥3", "Multiple": "×Multiple",
}

def _tok2type(t): return TUMOR_TYPES[t - OFFSET_TUMOR] if OFFSET_TUMOR <= t < OFFSET_POS else f"[{t}]"
def _tok2pos(p):  return POSITIONS[p - OFFSET_POS]     if OFFSET_POS  <= p < OFFSET_QTY else f"[{p}]"
def _tok2qty(q):  return QUANTITIES[q - OFFSET_QTY]    if OFFSET_QTY  <= q < OFFSET_QTY + len(QUANTITIES) else f"[{q}]"

def _lesion_dict(t, p, q):
    type_key = _tok2type(t)
    pos_key  = _tok2pos(p)
    qty_key  = _tok2qty(q)
    return {
        "type":     type_key,
        "type_lbl": _TYPE_DISPLAY.get(type_key, type_key),
        "pos":      pos_key,
        "pos_lbl":  _POS_DISPLAY.get(pos_key, pos_key),
        "qty":      qty_key,
        "qty_lbl":  _QTY_DISPLAY.get(qty_key, qty_key),
    }

_T5_LIVER_DESC = {
    "Fibrosis/Cirrhosis":   "cirrhotic liver morphology",
    "NoFibrosis/Cirrhosis": "normal liver parenchyma without fibrosis",
}
_T5_TUMOR_DESC = {
    "APHE_WO":   "arterial phase hyperenhancing lesion with portal-venous washout and capsule (LR-5, HCC pattern)",
    "RIM_ATYP":  "rim-enhancing lesion with atypical features (LR-M)",
    "APHE_NoWO": "arterial phase hyperenhancing lesion without definite washout (LR-3/4)",
    "HEM":       "T2-hyperintense lesion with peripheral nodular enhancement (hemangioma)",
    "CYST":      "simple hepatic cyst",
}
_T5_POSITION_DESC = {
    "L_LAT":      "left lobe lateral segment (segments 2/3)",
    "L_MED":      "left lobe medial segment (segment 4)",
    "R_ANT":      "right lobe anterior segment (segments 5/8)",
    "R_POST":     "right lobe posterior segment (segments 6/7)",
    "R_JUNCTION": "right lobe segmental junction",
    "CAUDATE":    "caudate lobe (segment 1)",
    "L_DIFFUSE":  "diffusely throughout the left lobe",
    "R_DIFFUSE":  "diffusely throughout the right lobe",
    "DIFFUSE":    "diffusely throughout the liver (bilateral)",
}
_T5_QUANTITY_DESC = {
    "1": "single lesion", "2": "two lesions",
    "GE3": "three or more lesions", "Multiple": "multiple/innumerable lesions",
}


def _structured_to_t5_input(sequence_raw: str) -> str:
    """Convert sequence_raw string → T5 prompt (matches finetune_report_gen.py format)."""
    parts = [p.strip() for p in sequence_raw.split(",") if p.strip()]
    if not parts:
        return "generate liver MRI report: No findings documented."
    liver_desc = _T5_LIVER_DESC.get(parts[0], parts[0])
    rest = parts[1:]
    lesion_parts = []
    if not rest:
        lesion_parts.append("No focal hepatic lesions identified.")
    else:
        i, n = 0, 1
        while i + 2 < len(rest):
            t, p, q = rest[i], rest[i+1], rest[i+2]
            lesion_parts.append(
                f"Lesion {n}: {_T5_TUMOR_DESC.get(t,t)}, "
                f"located in {_T5_POSITION_DESC.get(p,p)}, "
                f"{_T5_QUANTITY_DESC.get(q,q)}."
            )
            i += 3; n += 1
    return f"generate liver MRI report: Liver: {liver_desc}. {' '.join(lesion_parts)}"


def load_t5():
    """Load fine-tuned T5 model (non-blocking, called at startup)."""
    global _t5_model, _t5_tokenizer, _t5_info
    t5_dir = None
    for p in _T5_SEARCH_PATHS:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
            t5_dir = p
            break
    if t5_dir is None:
        _t5_info["error"] = "T5 checkpoint not found; template-only mode."
        print(f"[T5] {_t5_info['error']}")
        return
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        _t5_tokenizer = T5Tokenizer.from_pretrained(t5_dir)
        _t5_model     = T5ForConditionalGeneration.from_pretrained(t5_dir)
        _t5_model.eval()
        _t5_info["path"] = t5_dir
        print(f"[T5] Loaded from {t5_dir}")
    except Exception as e:
        _t5_info["error"] = str(e)
        print(f"[T5] Load failed: {e}")


def generate_t5_report(sequence_raw: str) -> str | None:
    """Generate NL report using T5; returns None if model not loaded."""
    if _t5_model is None or _t5_tokenizer is None:
        return None
    try:
        prompt = _structured_to_t5_input(sequence_raw)
        enc = _t5_tokenizer(prompt, return_tensors="pt",
                            max_length=256, truncation=True)
        with torch.no_grad():
            out = _t5_model.generate(
                enc["input_ids"], attention_mask=enc["attention_mask"],
                max_new_tokens=256, num_beams=4, early_stopping=True,
            )
        return _t5_tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return None


def _pair_f1(pred_pairs, gt_pairs):
    if not gt_pairs and not pred_pairs: return 1.0
    if not gt_pairs or not pred_pairs:  return 0.0
    tp = len(pred_pairs & gt_pairs)
    return 0.0 if tp == 0 else 2 * tp / (len(pred_pairs) + len(gt_pairs))

def run_val_inference():
    """Run model on entire val set and cache results."""
    global _val_cases, _val_manifest
    if _model is None:
        return
    cfg = Config()
    cfg.__post_init__()

    manifest_path = os.path.join(cfg.cache_root, cfg.cache_tag, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            all_recs = json.load(f)
        _val_manifest = [r for r in all_recs if r.get("split") == "val"]
    else:
        _val_manifest = []

    ds     = MRISeqDataset(cfg, split="val")
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    cases  = []
    idx    = 0
    with torch.no_grad():
        for batch_x, batch_y, _ in loader:
            batch_x = batch_x.to(_device)
            preds   = _model.generate(batch_x, max_len=batch_y.shape[1] + 2)
            for i in range(len(preds)):
                pred_info = parse_sequence(preds[i].cpu().numpy())
                gt_info   = parse_sequence(batch_y[i].numpy())

                gt_pairs   = {(t, p) for t, p, _ in gt_info["lesions"]}
                pred_pairs = {(t, p) for t, p, _ in pred_info["lesions"]}
                hit    = pred_pairs & gt_pairs
                missed = gt_pairs   - pred_pairs
                false  = pred_pairs - gt_pairs

                rec     = _val_manifest[idx] if idx < len(_val_manifest) else {}
                seq_raw = rec.get("sequence_raw", "")
                cases.append({
                    "idx":        idx,
                    "id":         rec.get("id", f"case_{idx}"),
                    "cache_path": rec.get("cache_path", ""),
                    "seq_raw":    seq_raw,
                    "report_t5":  generate_t5_report(seq_raw)  if seq_raw else None,
                    "report_llm": generate_llm_report(seq_raw) if seq_raw else None,
                    "gt":    [_lesion_dict(t, p, q) for t, p, q in gt_info["lesions"]],
                    "pred":  [_lesion_dict(t, p, q) for t, p, q in pred_info["lesions"]],
                    "hit":   [{"type": _tok2type(t), "type_lbl": _TYPE_DISPLAY.get(_tok2type(t), _tok2type(t)),
                               "pos": _tok2pos(p),  "pos_lbl":  _POS_DISPLAY.get(_tok2pos(p),  _tok2pos(p))}
                              for t, p in hit],
                    "missed":[{"type": _tok2type(t), "type_lbl": _TYPE_DISPLAY.get(_tok2type(t), _tok2type(t)),
                               "pos": _tok2pos(p),  "pos_lbl":  _POS_DISPLAY.get(_tok2pos(p),  _tok2pos(p))}
                              for t, p in missed],
                    "false": [{"type": _tok2type(t), "type_lbl": _TYPE_DISPLAY.get(_tok2type(t), _tok2type(t)),
                               "pos": _tok2pos(p),  "pos_lbl":  _POS_DISPLAY.get(_tok2pos(p),  _tok2pos(p))}
                              for t, p in false],
                    "pf1":      round(_pair_f1(pred_pairs, gt_pairs), 4),
                    "lesion1p": len(gt_pairs) > 0,
                    "gt_count": len(gt_pairs),
                })
                idx += 1
    _val_cases = sorted(cases, key=lambda c: -c["pf1"])
    print(f"[ValSet] Inference done: {len(_val_cases)} cases cached.")


def find_best_checkpoint():
    """Find best_model.pth by highest val_pair_f1_lesion1p_common in training_log.csv."""
    import csv
    pattern = os.path.join(PROJECT_ROOT, "runs", "exp_*", "best_model.pth")
    candidates = glob.glob(pattern)
    if not candidates:
        pattern2 = os.path.join(PROJECT_ROOT, "runs", "exp_*", "last_model.pth")
        candidates = glob.glob(pattern2)
    if not candidates:
        return None

    best_path, best_f1 = None, -1.0
    for ckpt in candidates:
        log_path = os.path.join(os.path.dirname(ckpt), "training_log.csv")
        if not os.path.exists(log_path):
            continue
        try:
            with open(log_path) as f:
                for row in csv.DictReader(f):
                    try:
                        v = float(row.get("val_pair_f1_lesion1p_common", ""))
                        if v > best_f1:
                            best_f1, best_path = v, ckpt
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

    # Fallback: alphabetically last if no log found
    return best_path if best_path else sorted(candidates)[-1]


def load_model():
    global _model, _device, _model_info
    ckpt_path = _model_info.get("path") or find_best_checkpoint()
    if ckpt_path is None:
        _model_info["error"] = "No checkpoint found in runs/. Please train the model first."
        return

    try:
        cfg = Config()
        # Auto-detect model_size from the checkpoint's config.json
        cfg_json = os.path.join(os.path.dirname(ckpt_path), "config.json")
        if os.path.exists(cfg_json):
            with open(cfg_json) as f:
                saved_cfg = json.load(f)
            cfg.model_size = saved_cfg.get("model_size", cfg.model_size)
        cfg.__post_init__()
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = MRIReportGenerator(cfg).to(_device)
        state = torch.load(ckpt_path, map_location=_device, weights_only=True)
        _model.load_state_dict(state, strict=True)
        _model.eval()
        _model_info["path"] = ckpt_path
        print(f"[Model] Loaded from: {ckpt_path}  device={_device}")
        import threading
        threading.Thread(target=run_val_inference, daemon=True).start()
    except Exception as e:
        _model_info["error"] = str(e)
        traceback.print_exc()


# ── Preprocessing ─────────────────────────────────────────────────────────────
_norm_transform = ScaleIntensityRangePercentiles(
    lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
)
_resize_transform = ResizeWithPadOrCrop(spatial_size=ROI_SIZE)


def preprocess_volume(path: str) -> torch.Tensor:
    """Load .nii.gz, normalize [0,1], resize to ROI_SIZE. Returns (1,H,W,D)."""
    data = nib.load(path).get_fdata().astype(np.float32)
    t = torch.tensor(data).unsqueeze(0)   # (1, H, W, D)
    t = _norm_transform(t)
    t = _resize_transform(t)
    return t                               # (1, 192, 192, 96)


def preprocess_mask(path: str) -> torch.Tensor:
    """Load mask, binarize, resize to ROI_SIZE. Returns (1,H,W,D)."""
    data = nib.load(path).get_fdata().astype(np.float32)
    data = (data > 0.5).astype(np.float32)
    t = torch.tensor(data).unsqueeze(0)   # (1, H, W, D)
    t = _resize_transform(t)
    t = (t > 0.5).float()
    return t


def make_input_tensor(art_path, pv_path, mask_path) -> torch.Tensor:
    """Returns (1, 3, H, W, D) ready for model."""
    art  = preprocess_volume(art_path)
    pv   = preprocess_volume(pv_path)
    mask = preprocess_mask(mask_path)
    x = torch.cat([art, pv, mask], dim=0)   # (3, H, W, D)
    return x.unsqueeze(0)                   # (1, 3, H, W, D)


# ── Token decoding ────────────────────────────────────────────────────────────
_LIVER_DISPLAY = {
    "Fibrosis/Cirrhosis":   "Fibrosis / Cirrhosis",
    "NoFibrosis/Cirrhosis": "No Fibrosis / No Cirrhosis",
}

_TUMOR_DISPLAY = {
    "APHE_WO":    "APHE with washout (APHE_WO)",
    "RIM_ATYP":   "Rim APHE / Atypical (RIM_ATYP)",
    "APHE_NoWO":  "APHE without washout (APHE_NoWO)",
    "HEM":        "Hemangioma (HEM)",
    "CYST":       "Cyst (CYST)",
}

_POS_DISPLAY = {
    "L_LAT":      "Left lobe – lateral",
    "L_MED":      "Left lobe – medial",
    "R_ANT":      "Right lobe – anterior",
    "R_POST":     "Right lobe – posterior",
    "R_JUNCTION": "Right lobe – junction",
    "CAUDATE":    "Caudate lobe",
    "L_DIFFUSE":  "Left lobe – diffuse",
    "R_DIFFUSE":  "Right lobe – diffuse",
    "DIFFUSE":    "Diffuse (bilateral)",
}

_QTY_DISPLAY = {
    "1": "Single (1)",
    "2": "Two (2)",
    "GE3": "Three or more (≥3)",
    "Multiple": "Multiple",
}


def decode_sequence(tokens: list) -> dict:
    """
    Convert token id list → structured report dict.
    Sequence format: [BOS, liver, type1, pos1, qty1, ..., EOS]
    """
    report = {"liver": None, "lesions": [], "no_lesion": False, "raw_tokens": tokens}

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in (PAD, BOS, EOS):
            i += 1
            continue
        if tok == NO_LESION:
            report["no_lesion"] = True
            i += 1
            continue
        if OFFSET_LIVER <= tok < OFFSET_TUMOR:
            report["liver"] = LIVER_TYPES[tok - OFFSET_LIVER]
            i += 1
            continue
        if OFFSET_TUMOR <= tok < OFFSET_POS:
            # Expect: type, position, quantity
            type_name = TUMOR_TYPES[tok - OFFSET_TUMOR]
            pos_name  = None
            qty_name  = None
            if i + 1 < len(tokens) and OFFSET_POS <= tokens[i + 1] < OFFSET_QTY:
                pos_name = POSITIONS[tokens[i + 1] - OFFSET_POS]
                if i + 2 < len(tokens) and OFFSET_QTY <= tokens[i + 2] < VOCAB_SIZE:
                    qty_name = QUANTITIES[tokens[i + 2] - OFFSET_QTY]
                    i += 3
                else:
                    i += 2
            else:
                i += 1
            report["lesions"].append({
                "type": type_name,
                "position": pos_name,
                "quantity": qty_name,
            })
            continue
        i += 1

    return report


_LIVER_SENTENCE = {
    "Fibrosis/Cirrhosis":   "The liver demonstrates morphologic features of cirrhosis.",
    "NoFibrosis/Cirrhosis": "The liver parenchyma appears within normal limits without evidence of cirrhosis.",
}

_TYPE_SENTENCE = {
    "APHE_WO":   "arterial phase hyperenhancement with washout appearance, consistent with LR-5 (HCC)",
    "RIM_ATYP":  "rim arterial phase hyperenhancement / atypical enhancement pattern (LR-M)",
    "APHE_NoWO": "arterial phase hyperenhancement without washout, indeterminate (LR-3/LR-4)",
    "HEM":       "T2 hyperintense lesion with peripheral nodular enhancement, consistent with hemangioma",
    "CYST":      "well-defined T2 hyperintense lesion without internal enhancement, consistent with a simple cyst",
}

_QTY_SENTENCE = {
    "1":        "a single",
    "2":        "two",
    "GE3":      "three or more",
    "Multiple": "multiple",
}

_POS_SENTENCE = {
    "L_LAT":      "in the left lobe (lateral segment)",
    "L_MED":      "in the left lobe (medial segment)",
    "R_ANT":      "in the right lobe (anterior segment)",
    "R_POST":     "in the right lobe (posterior segment)",
    "R_JUNCTION": "at the right lobe segmental junction",
    "CAUDATE":    "in the caudate lobe",
    "L_DIFFUSE":  "diffusely throughout the left lobe",
    "R_DIFFUSE":  "diffusely throughout the right lobe",
    "DIFFUSE":    "diffusely throughout the liver",
}

def report_to_text(report: dict) -> str:
    """Convert decoded report dict to radiology-style natural language."""
    lines = []

    liver = report.get("liver")
    lines.append(_LIVER_SENTENCE.get(liver, "Liver parenchyma evaluated."))

    lesions = report.get("lesions", [])
    if report.get("no_lesion") or not lesions:
        lines.append("No focal hepatic lesion is identified.")
    else:
        # Deduplicate by (type, position)
        seen = set()
        unique = []
        for les in lesions:
            key = (les.get("type"), les.get("position"))
            if key not in seen:
                seen.add(key)
                unique.append(les)

        findings = []
        for les in unique:
            t    = les.get("type", "")
            p    = les.get("position", "")
            q    = les.get("quantity", "")
            tdesc = _TYPE_SENTENCE.get(t, t)
            pdesc = _POS_SENTENCE.get(p, "in the liver")
            qdesc = _QTY_SENTENCE.get(q, "")
            qty_str = f"{qdesc} focus" if qdesc else "focus/foci"
            if q in ("GE3", "Multiple"):
                qty_str = f"{qdesc} foci"
            findings.append(f"There is {qty_str} {pdesc} demonstrating {tdesc}.")

        lines.extend(findings)

    lines.append("Hepatic vasculature is patent.")
    return "\n\n".join(lines)


# ── Global JSON error handlers ────────────────────────────────────────────────
@app.errorhandler(404)
def err_404(e):
    return jsonify({"error": f"404 Not Found: {request.path}"}), 404

@app.errorhandler(405)
def err_405(e):
    return jsonify({"error": f"405 Method Not Allowed: {request.method} {request.path}"}), 405

@app.errorhandler(413)
def err_413(e):
    return jsonify({"error": "File too large (max 512 MB per request)"}), 413

@app.errorhandler(500)
def err_500(e):
    return jsonify({"error": f"Internal server error: {e}"}), 500


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    model_status = "ready" if _model is not None else "unavailable"
    model_path = _model_info.get("path", "")
    model_error = _model_info.get("error", "")
    return render_template(
        "index.html",
        model_status=model_status,
        model_path=model_path,
        model_error=model_error,
    )


@app.route("/status")
def status():
    return jsonify({
        "model_loaded": _model is not None,
        "checkpoint":   _model_info.get("path"),
        "error":        _model_info.get("error"),
        "device":       str(_device) if _device else None,
        "llm_ready":    _llm_client is not None,
        "llm_model":    _llm_info.get("model"),
        "llm_error":    _llm_info.get("error"),
        "t5_ready":     _t5_model is not None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if _model is None:
        return jsonify({"error": f"Model not loaded. {_model_info.get('error', '')}"}), 503

    # ── Save uploaded files ───────────────────────────────────────────────────
    tmp_files = []
    try:
        required = ["art", "pv", "mask"]
        for key in required:
            if key not in request.files or request.files[key].filename == "":
                return jsonify({"error": f"Missing file: {key}"}), 400

        paths = {}
        for key in required:
            f = request.files[key]
            suffix = ".nii.gz" if f.filename.endswith(".nii.gz") else ".nii"
            tmp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}{suffix}")
            f.save(tmp_path)
            tmp_files.append(tmp_path)
            paths[key] = tmp_path

        # ── Preprocess ────────────────────────────────────────────────────────
        x = make_input_tensor(paths["art"], paths["pv"], paths["mask"])
        x = x.to(_device)

        # Cache the preprocessed volume for 3-axis slice serving
        upload_id = str(uuid.uuid4())
        _upload_vols[upload_id] = x[0].cpu().numpy()   # (3, H, W, D)
        # Keep only the 5 most recent uploads to avoid memory leak
        if len(_upload_vols) > 5:
            oldest_key = next(iter(_upload_vols))
            del _upload_vols[oldest_key]

        # ── Inference ─────────────────────────────────────────────────────────
        with torch.no_grad():
            tokens = _model.generate(x, max_len=MAX_GENERATE_LEN)

        tokens_list = tokens[0].cpu().tolist()

        # ── Decode ────────────────────────────────────────────────────────────
        report    = decode_sequence(tokens_list)
        text      = report_to_text(report)
        seq_raw   = report.get("_seq_raw", "")   # built below from decoded report
        # Reconstruct sequence_raw string from decoded report for T5 input
        _liver_raw = report.get("liver") or ""
        _les_parts = []
        for _l in report.get("lesions", []):
            _les_parts += [_l.get("type",""), _l.get("position",""), _l.get("quantity","")]
        _seq_raw_str = ", ".join([_liver_raw] + _les_parts) if _liver_raw else ""
        t5_text  = generate_t5_report(_seq_raw_str) if _seq_raw_str else None
        llm_text = generate_llm_report(_seq_raw_str, text) if _seq_raw_str else None

        _, _, h, w, d = x.shape   # x is (1, 3, H, W, D)

        # Pre-render mid-slices for all 3 axes (ART channel) as base64
        def mid_b64(vol3, axis):
            """vol3: (3,H,W,D) numpy. axis: 0=sag,1=cor,2=ax"""
            ch = vol3[0]   # ART channel
            n  = ch.shape[axis]
            mid = n // 2
            if axis == 0: sl = ch[mid, :, :]
            elif axis == 1: sl = ch[:, mid, :]
            else:           sl = ch[:, :, mid]
            buf = _slice_to_png(np.rot90(sl))
            return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

        vol_np = _upload_vols[upload_id]
        # Return mid axial slice for all 3 channels so they show immediately
        def mid_b64_ch(vol3, ch, axis):
            c = vol3[ch]
            n = c.shape[axis]; mid = n // 2
            if axis == 0: sl = c[mid, :, :]
            elif axis == 1: sl = c[:, mid, :]
            else:           sl = c[:, :, mid]
            buf = _slice_to_png(np.rot90(sl))
            return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

        mid_slices = {
            "art":  mid_b64_ch(vol_np, 0, 2),
            "pv":   mid_b64_ch(vol_np, 1, 2),
            "mask": mid_b64_ch(vol_np, 2, 2),
        }

        # Try to extract case ID from the uploaded filename to look up GT
        import re as _re
        gt_case = None
        art_fname = request.files["art"].filename
        m = _re.search(r'(Liv_[A-Za-z0-9]+)', art_fname)
        val_case_id = None
        if m:
            val_case_id = m.group(1)  # raw ID for client-side polling
            if _val_cases:
                cid = val_case_id.upper()
                gt_case = next((c for c in _val_cases
                                if c["id"].upper() == cid), None)

        return jsonify({
            "success":    True,
            "upload_id":  upload_id,
            "vol_shape":  {"h": int(h), "w": int(w), "d": int(d)},
            "mid_slices": mid_slices,
            "report_text":     text,
            "report_text_t5":  t5_text,
            "report_text_llm": llm_text,
            "report": {
                "liver":    report["liver"],
                "no_lesion": report["no_lesion"],
                # Deduplicate by (type, position) — model sometimes repeats tokens
                "lesions":  list({(l["type"], l["position"]): l
                                  for l in report["lesions"]}.values()),
            },
            "tokens": tokens_list,
            "gt": gt_case,          # populated if val inference already done
            "val_case_id": val_case_id,  # ID for client to poll /api/gt/<id>
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        for p in tmp_files:
            try:
                os.remove(p)
            except Exception:
                pass


# ── Val set routes ────────────────────────────────────────────────────────────
@app.route("/valset")
def valset_page():
    if _val_cases is None:
        if _model is None:
            return "<h2>Model not loaded.</h2>", 503
        return ("<h2 style='font-family:sans-serif;padding:40px;color:#aaa'>"
                "⏳ Val set inference running in background… "
                "Refresh in ~2 minutes.</h2>"), 202
    lesion1p = [c for c in _val_cases if c["lesion1p"]]
    noles    = [c for c in _val_cases if not c["lesion1p"]]
    hits     = sum(1 for c in lesion1p if c["hit"])
    noles_ok = sum(1 for c in [c for c in _val_cases if not c["lesion1p"]] if not c["pred"])
    mean_f1  = round(float(np.mean([c["pf1"] for c in lesion1p])), 4) if lesion1p else 0
    return render_template(
        "valset.html",
        cases        = _val_cases,
        stats = {
            "total":       len(_val_cases),
            "lesion1p":    len(lesion1p),
            "noles":       len(noles),
            "hits":        hits,
            "mean_f1":     mean_f1,
            "noles_ok":    noles_ok,
            "les_missed":  sum(1 for c in lesion1p if not c["pred"]),
        },
        model_path = _model_info.get("path", ""),
    )

@app.route("/api/valset")
def api_valset():
    if _val_cases is None:
        run_val_inference()
    return jsonify(_val_cases or [])

@app.route("/api/gt/<case_id>")
def api_gt(case_id):
    """Return GT info for a case by ID.
    If full val cache is ready, look up from it.
    Otherwise run inference just for this one case (fast, <1s)."""
    cid = case_id.upper()

    # Fast path: full cache already done
    if _val_cases is not None:
        case = next((c for c in _val_cases if c["id"].upper() == cid), None)
        if case is None:
            return jsonify({"status": "not_found"}), 404
        return jsonify({"status": "ok", "gt": case})

    # Slow path: find & run just this one case from val manifest
    if _model is None:
        return jsonify({"status": "error", "msg": "model not loaded"}), 503

    cfg = Config(); cfg.__post_init__()
    manifest_path = os.path.join(cfg.cache_root, cfg.cache_tag, "manifest.json")
    if not os.path.exists(manifest_path):
        return jsonify({"status": "not_found"}), 404

    # manifest lives at <base>/data_cache/summary/manifest.json
    # cache_path entries are ./data_cache/summary/xxx.npz relative to <base>
    manifest_dir = os.path.dirname(manifest_path)          # .../data_cache/summary
    cache_base   = os.path.dirname(os.path.dirname(manifest_dir))  # <base>

    with open(manifest_path) as f:
        all_recs = json.load(f)
    rec = next((r for r in all_recs
                if r.get("split") == "val" and r.get("id", "").upper() == cid), None)
    if rec is None:
        return jsonify({"status": "not_found"}), 404

    # Load npz, run inference on this single case
    path = rec.get("cache_path", "")
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(cache_base, path))
    data = np.load(path)
    x = torch.tensor(data["x"]).unsqueeze(0).to(_device)   # (1,3,H,W,D)
    y = torch.tensor(data["y"])                              # (seq_len,)

    with torch.no_grad():
        pred_tok = _model.generate(x, max_len=y.shape[0] + 2)

    pred_info = parse_sequence(pred_tok[0].cpu().numpy())
    gt_info   = parse_sequence(y.numpy())

    gt_pairs   = {(t, p) for t, p, _ in gt_info["lesions"]}
    pred_pairs = {(t, p) for t, p, _ in pred_info["lesions"]}
    hit    = pred_pairs & gt_pairs
    missed = gt_pairs   - pred_pairs
    false  = pred_pairs - gt_pairs

    case = {
        "id":     rec["id"],
        "gt":     [_lesion_dict(t, p, q) for t, p, q in gt_info["lesions"]],
        "pred":   [_lesion_dict(t, p, q) for t, p, q in pred_info["lesions"]],
        "hit":    [{"type": _tok2type(t), "pos": _tok2pos(p)} for t, p in hit],
        "missed": [{"type": _tok2type(t), "pos": _tok2pos(p)} for t, p in missed],
        "false":  [{"type": _tok2type(t), "pos": _tok2pos(p)} for t, p in false],
        "pf1":    round(_pair_f1(pred_pairs, gt_pairs), 4),
        "lesion1p": len(gt_pairs) > 0,
    }
    return jsonify({"status": "ok", "gt": case})

def _load_vol(case_id):
    """Return (vol, path) where vol is (3,H,W,D), or (None, None).
    Checks upload cache first, then val-set cache."""
    # Check in-memory upload cache
    if case_id in _upload_vols:
        return _upload_vols[case_id], None
    # Check val-set manifest
    case = next((c for c in (_val_cases or []) if c["id"] == case_id), None)
    if case is None:
        return None, None
    path = case.get("cache_path", "")
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    return data["x"], path   # (3, H, W, D)

def _slice_to_png(sl_2d):
    """Normalise a 2D float array → PNG bytes."""
    vmin, vmax = sl_2d.min(), sl_2d.max()
    if vmax > vmin:
        sl_2d = (sl_2d - vmin) / (vmax - vmin)
    arr = (sl_2d * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

@app.route("/vol_shape/<case_id>")
def vol_shape(case_id):
    """Return {h,w,d} of the volume."""
    vol, _ = _load_vol(case_id)
    if vol is None:
        return jsonify({}), 404
    _, h, w, d = vol.shape
    return jsonify({"h": h, "w": w, "d": d})

@app.route("/mri_slice/<case_id>/<int:channel>")
@app.route("/mri_slice/<case_id>/<int:channel>/<int:axis>/<int:sidx>")
def mri_slice(case_id, channel, axis=2, sidx=-1):
    """Return a 2D PNG slice from the cached .npz.
    axis: 0=sagittal(YZ), 1=coronal(XZ), 2=axial(XY)
    sidx: slice index (-1 = mid)
    """
    try:
        vol, _ = _load_vol(case_id)
        if vol is None:
            return "", 404
        ch = vol[min(channel, 2)]      # (H, W, D)
        sizes = ch.shape               # (H, W, D)
        n = sizes[axis]
        idx = n // 2 if sidx < 0 or sidx >= n else sidx
        if axis == 0:   sl = ch[idx, :, :]   # sagittal  → (W, D)
        elif axis == 1: sl = ch[:, idx, :]   # coronal   → (H, D)
        else:           sl = ch[:, :, idx]   # axial     → (H, W)
        buf = _slice_to_png(np.rot90(sl))
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return str(e), 500

@app.route("/bulk_slices/<case_id>/<int:channel>/<int:axis>")
def bulk_slices(case_id, channel, axis):
    """Return ALL slices along one axis as JSON array of base64 PNGs."""
    try:
        vol, _ = _load_vol(case_id)
        if vol is None:
            return jsonify([]), 404
        ch = vol[min(channel, 2)]   # (H, W, D)
        n  = ch.shape[axis]
        out = []
        for i in range(n):
            if axis == 0:   sl = ch[i, :, :]
            elif axis == 1: sl = ch[:, i, :]
            else:           sl = ch[:, :, i]
            buf = _slice_to_png(np.rot90(sl))
            out.append(base64.b64encode(buf.read()).decode())
        return jsonify(out)
    except Exception as e:
        return str(e), 500

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint path (default: auto-detect latest run)")
    args = parser.parse_args()

    if args.checkpoint:
        _model_info["path"] = args.checkpoint

    load_model()
    load_t5()
    init_llm()
    # ProxyFix: needed when accessed through OOD reverse proxy (/rnode/host/port/)
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1)
    app.run(host=args.host, port=args.port, debug=False)
