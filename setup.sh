#!/bin/bash
# ============================================================
#  MRI2Rep — Environment Setup Script
#  Run once on a new server to create the Python environment.
#
#  Usage:
#    bash setup.sh              # creates conda env named "mri2rep"
#    bash setup.sh myenv        # custom env name
# ============================================================

ENV_NAME=${1:-mri2rep}
CUDA_VERSION=${CUDA_VERSION:-"cu124"}   # adjust to match your CUDA (cu118, cu121, cu124 …)

echo "======================================================"
echo "  MRI2Rep environment setup"
echo "  Conda env  : $ENV_NAME"
echo "  CUDA target: $CUDA_VERSION"
echo "======================================================"

# ── 1. Create conda environment ──────────────────────────────────────────────
conda create -y -n "$ENV_NAME" python=3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── 2. Install PyTorch (CUDA-aware) ──────────────────────────────────────────
# Check https://pytorch.org/get-started/locally/ for the exact URL for your CUDA.
pip install torch==2.5.1 torchvision \
    --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"

# ── 3. Install remaining dependencies ────────────────────────────────────────
pip install -r requirements.txt

# ── 4. Download required NLTK data ───────────────────────────────────────────
python -c "
import nltk
for pkg in ['wordnet', 'omw-1.4', 'punkt', 'punkt_tab']:
    nltk.download(pkg, quiet=True)
print('NLTK data downloaded.')
"

# ── 5. Verify ─────────────────────────────────────────────────────────────────
python -c "
import torch, monai, nibabel, matplotlib, nltk
print(f'torch   : {torch.__version__}  (CUDA: {torch.version.cuda})')
print(f'monai   : {monai.__version__}')
print(f'nibabel : {nibabel.__version__}')
print(f'GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no GPU\"}')
print('All imports OK.')
"

echo ""
echo "Done! Activate the environment with:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Next step — verify data and start training:"
echo "  python main.py"
