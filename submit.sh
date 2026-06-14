#!/bin/bash
# ============================================================
#  MRI2Rep — SLURM Submit Script
#
#  Usage:
#    sbatch submit.sh                        # fresh training
#    sbatch submit.sh --resume runs/exp_...  # resume from checkpoint
#
#  Tune the #SBATCH lines below for your cluster:
#    --partition  : the GPU partition name on your HPC
#    --account    : your project/allocation account (remove if not needed)
#    --gres       : GPU type/count (e.g. gpu:a100:1, gpu:h100:1, gpu:1)
#    --mem        : system RAM (64G recommended for data loading)
#    --time       : wall time limit
# ============================================================

#SBATCH --job-name=MRI2Rep
#SBATCH --partition=gpu
#SBATCH --account=pi_lhs7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -e
mkdir -p logs

# ── Activate environment ──────────────────────────────────────────────────────
# Option A: conda
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate lxr
# Option B: venv (uncomment if using venv instead)
# source /path/to/venv/bin/activate

# ── Diagnostics ──────────────────────────────────────────────────────────────
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "CPUs     : $SLURM_CPUS_PER_TASK"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Python   : $(python --version)"
echo "Start    : $(date)"
echo "Work dir : $(pwd)"

# ── Run training ──────────────────────────────────────────────────────────────
python main.py "$@"

echo "Done: $(date)"
