from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    # ---- Model size switch: change this one line to swap everything ----
    model_size: str = "large"   # "large" | "small"

    # ---- Paths (image roots not needed — data is fully preprocessed into .npz) ----
    art_img_root: str = ""
    pv_img_root: str = ""
    mask_root: str = ""
    art_img_pattern: str = "processed_{id}_art_img.nii.gz"
    pv_img_pattern: str = "processed_{id}_pv_img.nii.gz"
    mask_pattern: str = "processed_{id}_art_seg.nii.gz"
    outdir: str = "./runs"
    pretrained_backbone: str = ""   # set by __post_init__ based on model_size

    # ---- Data cache ----
    cache_root: str = "/nfs/roberts/project/pi_lhs7/xl754/data_cache"
    label_csv: str = ""
    cache_tag: str = "summary"

    # ---- Data & Preprocessing ----
    roi_size: Tuple[int, int, int] = (192, 192, 96)
    max_lesions: int = 8
    max_seq_len: int = 1 + 1 + (3 * 8) + 1  # = 27

    # ---- Training ----
    epochs: int = 1000
    batch_size: int = 24
    lr: float = 5e-5
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    num_workers: int = 8
    seed: int = 42
    use_class_weights: bool = True
    class_weight_beta: float = 0.999
    rare_sampling_ratio: float = 0.4
    no_lesion_ratio: float = 0.2
    label_smoothing: float = 0.1
    early_stop_patience: int = 150
    word_dropout_p: float = 0.15
    use_amp: bool = True
    aux_weight: float = 0.5

    # ---- Best-model checkpoint guard ----
    min_epoch_best: int = 15
    min_lesion1p_f1_for_best: float = 0.02

    # ---- Scheduled Sampling ----
    ss_max_prob: float = 0.3
    ss_start_epoch: int = 10
    ss_warmup_epochs: int = 20

    # ---- Model architecture (derived from model_size in __post_init__) ----
    d_model: int = 0
    nheads: int = 8
    enc_layers: int = 4
    dec_layers: int = 6
    dim_feedforward: int = 0
    dropout: float = 0.3
    backbone_base: int = 0
    vis_pool_size: Tuple[int, int, int] = (8, 8, 4)

    def __post_init__(self):
        if self.model_size == "large":
            self.d_model             = 256
            self.backbone_base       = 48
            self.dim_feedforward     = 1024
            self.pretrained_backbone = "./pretrained/backbone_pv_pretrain_large.pth"
        else:  # "small"
            self.d_model             = 128
            self.backbone_base       = 32
            self.dim_feedforward     = 512
            self.pretrained_backbone = "./pretrained/backbone_pv_pretrain.pth"


# ================= VOCABULARY DEFINITION =================
# Special Tokens
PAD = 0
BOS = 1
EOS = 2
NO_LESION = 3

# Categories
LIVER_TYPES = ["Fibrosis/Cirrhosis", "NoFibrosis/Cirrhosis"]
# Type priority: higher clinical signal first
TUMOR_TYPES = ["APHE_WO", "RIM_ATYP", "APHE_NoWO", "HEM", "CYST"]
# Position priority: focal first, diffuse last
POSITIONS   = ["L_LAT", "L_MED", "R_ANT", "R_POST", "R_JUNCTION", "CAUDATE", "L_DIFFUSE", "R_DIFFUSE", "DIFFUSE"]
# Quantity buckets
QUANTITIES  = ["1", "2", "GE3", "Multiple"]

# Offset mappings to build a single vocab
# Vocab structure: [Specials] + [Liver] + [Tumor] + [Position] + [Quantity]
OFFSET_LIVER = 4
OFFSET_TUMOR = OFFSET_LIVER + len(LIVER_TYPES)
OFFSET_POS   = OFFSET_TUMOR + len(TUMOR_TYPES)
OFFSET_QTY   = OFFSET_POS + len(POSITIONS)
VOCAB_SIZE   = OFFSET_QTY + len(QUANTITIES)

def get_token_type(token_id: int) -> str:
    if token_id in [PAD, BOS, EOS, NO_LESION]: return "special"
    if OFFSET_LIVER <= token_id < OFFSET_TUMOR: return "liver"
    if OFFSET_TUMOR <= token_id < OFFSET_POS: return "tumor"
    if OFFSET_POS <= token_id < OFFSET_QTY: return "pos"
    if OFFSET_QTY <= token_id < VOCAB_SIZE: return "qty"
    return "unknown"
