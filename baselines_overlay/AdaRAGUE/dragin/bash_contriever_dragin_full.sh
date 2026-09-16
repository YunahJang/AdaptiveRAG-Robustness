#!/bin/bash
#SBATCH --job-name  contriever_dragin_full
#SBATCH --time      7-00:00:00
#SBATCH -c          4
#SBATCH --gpus      2
#SBATCH --nodelist a6k01

# ============================================
# Contriever DRAGIN experiments
# Datasets: 2wikimultihopqa, squad
# Models: QwQ-32B, Llama-3.1-8B
# All variations (10 types)
# ============================================


# Project root (override: export QR_ROOT=/path/to/QueryRobustness)
QR_ROOT="${QR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

# Activate your conda installation before this line.
cd ${QR_ROOT}/baselines/AdaRAGUE/dragin

conda activate dragin_llama

echo "============================================"
echo "🚀 Contriever DRAGIN Experiments (Full)"
echo "============================================"
echo "Datasets: 2wikimultihopqa, squad"
echo "Models: QwQ-32B-Preview, Llama-3.1-8B-Instruct"
echo "Variations: 10 types"
echo "Start Time: $(date)"
echo "Node: $(hostname -s)"
echo "============================================"
echo ""

# Generate the config files (if not done yet)
if [ ! -d "./configs_contriever/2wikimultihopqa" ]; then
    echo "[INFO] Generating Contriever configs..."
    python ./configs_contriever/generate_configs.py
    echo ""
fi

# datasets
DATASETS=("2wikimultihopqa" "squad")

# all variations
VARIATIONS=("formality" "readability" "declarative" "imperative" "grammar_error" "keyboard_error" "ocr_error")

# models
MODELS=(
    "Qwen/QwQ-32B"
    "meta-llama/Llama-3.1-8B-Instruct"
)

# run for every model
for model_path in "${MODELS[@]}"; do
    model_name=$(basename "$model_path" | tr '[:upper:]' '[:lower:]')

    echo ""
    echo "════════════════════════════════════════════"
    echo "🤖 Model: $model_name"
    echo "════════════════════════════════════════════"
    echo ""

    # run for every dataset
    for dataset in "${DATASETS[@]}"; do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 Dataset: $dataset"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # run for every variation
        for var in "${VARIATIONS[@]}"; do
            config_file="./configs_contriever/${dataset}/DRAGIN_${var}.json"

            if [ ! -f "$config_file" ]; then
                echo "⚠️  Config not found: $config_file"
                continue
            fi

            echo "[$(date +%H:%M:%S)] Running: $dataset / DRAGIN / $var / $model_name"

            # temporarily override model_name_or_path in the config
            temp_config="/tmp/dragin_config_${dataset}_${var}_${model_name}.json"
            jq --arg model "$model_path" '.model_name_or_path = $model' "$config_file" > "$temp_config"

            if python ./src/main_newstore.py -c "$temp_config"; then
                echo "  ✅ Success"
            else
                echo "  ❌ Failed"
            fi

            # remove the temporary file
            rm -f "$temp_config"
        done

        echo "✅ $dataset completed for $model_name!"
        echo ""
    done
done

echo ""
echo "============================================"
echo "✅ All experiments completed!"
echo "End Time: $(date)"
echo "============================================"
echo ""
echo "📁 Results saved to: ./results_contriever/"
echo ""
echo "Summary:"
echo "  - Datasets: ${#DATASETS[@]}"
echo "  - Variations: ${#VARIATIONS[@]}"
echo "  - Models: ${#MODELS[@]}"
echo "  - Total runs: $((${#DATASETS[@]} * ${#VARIATIONS[@]} * ${#MODELS[@]}))"
echo "============================================"
