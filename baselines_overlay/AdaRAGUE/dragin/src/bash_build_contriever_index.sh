#!/bin/bash
#SBATCH --job-name  contriever_index
#SBATCH --time      2-00:00:00
#SBATCH -c          4
#SBATCH --gpus      1
#SBATCH --mem       100G
#SBATCH --nodelist  a6k01


# Project root (override: export QR_ROOT=/path/to/QueryRobustness)
QR_ROOT="${QR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"

# Activate your conda installation here, e.g.:
#   source "$(conda info --base)/etc/profile.d/conda.sh"

cd ${QR_ROOT}/baselines/AdaRAGUE/dragin/src
conda activate dragin

python build_contriever_index.py