#!/bin/bash
#SBATCH --job-name  contriever_single
#SBATCH --time      1-00:00:00
#SBATCH -c          4
#SBATCH --gpus      1
#SBATCH --mem       100G

# ========================================
# Example: run and evaluate a single config
# Usage: ./run_single_example.sh [config_file]
# ========================================


# Project root (override: export QR_ROOT=/path/to/QueryRobustness)
QR_ROOT="${QR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"

# Activate your conda installation here, e.g.:
#   source "$(conda info --base)/etc/profile.d/conda.sh"

cd ${QR_ROOT}/baselines/AdaRAGUE/dragin
conda activate dragin

# Config file (default, or passed as an argument)
CONFIG_FILE=${1:-"./configs_contriever/SR-RAG_original.json"}

echo "========================================"
echo "Running Contriever-based RAG"
echo "Config: $CONFIG_FILE"
echo "========================================"

# 1. Run inference
python ./src/main_newstore.py -c "$CONFIG_FILE"

# 2. Read the information back from the config
CONFIG_NAME=$(basename "$CONFIG_FILE" .json)
METHOD=$(echo "$CONFIG_NAME" | cut -d'_' -f1 | tr '[:upper:]' '[:lower:]' | tr '-' '-')
VARIATION=$(echo "$CONFIG_NAME" | cut -d'_' -f2- | tr '[:upper:]' '[:lower:]')

# 3. Find the output directory (most recently created)
OUTPUT_BASE=$(python -c "
import json
with open('$CONFIG_FILE') as f:
    cfg = json.load(f)
    print(cfg['output_dir'])
" 2>/dev/null)

if [ -z "$OUTPUT_BASE" ]; then
    echo "[ERROR] Could not extract output_dir from config"
    exit 1
fi

# convert to an absolute path
if [[ "$OUTPUT_BASE" != /* ]]; then
    OUTPUT_BASE="${QR_ROOT}/baselines/AdaRAGUE/dragin/$OUTPUT_BASE"
fi

# find the most recently created directory
LATEST_DIR=$(ls -dt "${OUTPUT_BASE}/${METHOD}_${VARIATION}"* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "[ERROR] Output directory not found: ${OUTPUT_BASE}/${METHOD}_${VARIATION}*"
    exit 1
fi

echo "========================================"
echo "Evaluating results"
echo "Output Dir: $LATEST_DIR"
echo "========================================"

# 4. Run the evaluation
python ./src/evaluate.py --dir "$LATEST_DIR"

echo "========================================"
echo "DONE!"
echo "Results saved to: $LATEST_DIR"
echo "========================================"

# print the result
echo ""
echo "=== Results Summary ==="
cat "$LATEST_DIR/result.tsv"
