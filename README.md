# MRI2Rep

A 3D CNN + Transformer model that generates structured liver lesion reports from 3-channel MRI volumes (Arterial, Portal Venous, and Organ Mask).

The full pipeline has two stages: (1) extract structured labels from raw radiology reports using an LLM, then (2) train and evaluate the MRI2Rep model on the resulting dataset.

---

## Stage 1 — Label Extraction (`Report_Process/`)

A 3-step LLM pipeline (organ check → verification → tumor extraction) that converts free-text radiology reports into structured lesion label sequences.

**Setup:** set your API key and place raw `.txt` reports under `Report_Process/reports/all_splits/<split>/`.

```bash
cd Report_Process
export DEEPSEEK_API_KEY="your_api_key"
```

**Run per batch:**
```bash
SPLIT_DIRNAME="1_100" python main.py       # LLM extraction (3 consistency rounds)
SPLIT_DIRNAME="1_100" python post.py       # format normalization
# repeat for each batch: 101_200, 201_300, ...
```

**Aggregate results:**
```bash
python recover_2of3.py                              # recover 2/3-agreement cases, output rerun_ids.txt
python rerun_failed.py --rerun_txt ./logs/rerun_ids.txt --rounds 5   # re-run low-confidence cases
python build_accurate_dataset.py                    # → logs/accurate_summary.csv
python merge_summary_yes.py                         # merge all batch summaries
```

Final output `logs/accurate_summary.csv` is used as the label file for model training.

> `logs/` in this repo contains results for the first 1,000 cases only.

---

## Stage 2 — Model Training

### Setup

```bash
bash setup.sh
conda activate mri2rep
```

Configure the data path in `src/config.py`:
```python
cache_root = "/path/to/data_cache"   # directory containing .npz MRI files
cache_tag  = "summary"               # subfolder name
```

### Train

```bash
python pretrain_pv.py                          # optional: self-supervised backbone pretraining
python main.py                                 # train from scratch
python main.py --resume runs/exp_YYYYMMDD_XXX  # resume from checkpoint
sbatch submit.sh                               # SLURM submission
```

Checkpoints, training logs, and loss curves are saved to `runs/exp_<timestamp>/`.

### Evaluate & Visualize

```bash
# NLG metrics: BLEU / ROUGE / METEOR / BERTScore
python evaluate_nlg.py --run_dir runs/exp_XXX

# Per-case GT vs. prediction comparison
python analyze_preds.py --exp runs/exp_XXX --split val

# GradCAM attribution maps on MRI slices
python gradcam_vis.py --run_dir runs/exp_XXX
python gradcam_vis.py --run_dir runs/exp_XXX --case_id Liv_XXXXXXXX
```

---

## Model Architecture

```
Input (B, 3, 192, 192, 96)   ← [ART, PV, MASK]
  ↓
ConvBackbone3D (base=48)      →  (B, 256, 24, 24, 12)
  ↓
Visual TransformerEncoder (4 layers, d_model=256, 8 heads)
  ↓
TransformerDecoder (6 layers, cross-attention over 256 visual tokens)
  ↓
Linear → logits (B, L, 24 vocab)
```

**Output vocab (24 tokens):** `PAD / BOS / EOS / NO_LESION` · liver status (2) · tumor type (5) · position (9) · quantity (4)

---

## Requirements

```
torch>=2.1  monai>=1.2  nibabel>=5.0
numpy  pandas  matplotlib  tqdm
nltk  rouge-score  bert-score
```

See `requirements.txt` for pinned versions.
