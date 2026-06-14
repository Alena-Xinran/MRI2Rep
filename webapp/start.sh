#!/bin/bash
# ============================================================
#  Start MRI2Rep Web App
#
#  Usage:
#    bash webapp/start.sh               # port 5050
#    bash webapp/start.sh --port 8080   # custom port
#
#  Access from local machine via SSH tunnel:
#    ssh -L 5050:localhost:5050 <your-login-node>
#    Then open http://localhost:5050 in browser
#
#  Pass --checkpoint path/to/best_model.pth to override
#  the auto-detected checkpoint.
# ============================================================

set -e
cd "$(dirname "$0")/.."   # go to project root

source "$(conda info --base)/etc/profile.d/conda.sh"
# Try lxr first, fall back to UNA (which has torch)
conda activate lxr 2>/dev/null || conda activate UNA

echo "Starting MRI2Rep Web App..."
echo "Project root: $(pwd)"
python -m webapp.app "$@"
