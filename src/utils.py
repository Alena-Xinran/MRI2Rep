import os
import random
import numpy as np
import torch
import hashlib

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def stable_split(id_str: str, train=80, val=10, test=10) -> str:
    h = int(hashlib.md5(id_str.encode("utf-8")).hexdigest(), 16) % 100
    if h < train: return "train"
    if h < train + val: return "val"
    return "test"

def sort_lesions(lesions):
    """Sort lesions to ensure deterministic order (by Position then Type)."""
    # Accept either (t, p) or (t, p, q) tuples. Quantity does not affect ordering.
    def key_fn(x):
        if len(x) == 3:
            t, p, q = x
            return (p, t)
        t, p = x
        return (p, t)

    return sorted(lesions, key=key_fn)
