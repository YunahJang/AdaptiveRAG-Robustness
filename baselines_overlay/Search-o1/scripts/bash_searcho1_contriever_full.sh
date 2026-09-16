#!/bin/bash
#SBATCH --job-name  searcho1_contriever_full
#SBATCH --time      7-00:00:00
#SBATCH -c          4
#SBATCH --gpus      2
#SBATCH --nodelist a6k01

# ============================================
# Search-o1 with Contriever
# Datasets: 2wikimultihopqa, squad
# Models: QwQ-32B, Llama-3.1-8B
# All variations (10 types)
# ============================================


# Project root (override: export QR_ROOT=/path/to/QueryRobustness)
QR_ROOT="${QR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

cd ${QR_ROOT}/baselines/Search-o1/scripts

echo "============================================"
echo "🚀 Search-o1 with Contriever (Full)"
echo "============================================"
echo "Datasets: 2wikimultihopqa, squad"
echo "Models: QwQ-32B-Preview, Llama-3.1-8B-Instruct"
echo "Variations: 10 types"
echo "Start Time: $(date)"
echo "Node: $(hostname -s)"
echo "============================================"
echo ""

# Config
CONTRIEVER_INDEX="${QR_ROOT}/indexes/contriever_psgs_w100_fp16_stream/index.faiss"
CONTRIEVER_META="${QR_ROOT}/indexes/contriever_psgs_w100_fp16_stream/meta.jsonl"
DATA_BASE="${QR_ROOT}/processed_data"

# Datasets and variations
DATASETS=("2wikimultihopqa" "squad")
VARIATIONS=("original" "human" "formality" "readability" "declarative" "imperative" "grammar_error" "spelling_error" "keyboard_error" "ocr_error")

# Models
MODELS=(
    "Qwen/QwQ-32B-Preview"
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

        # Map dataset name for script argument
        if [ "$dataset" = "2wikimultihopqa" ]; then
            ds_arg="2wikimultihopqa"
        else
            ds_arg="$dataset"
        fi

        # run for every variation
        for var in "${VARIATIONS[@]}"; do
            if [ "$var" = "original" ]; then
                data_file="${DATA_BASE}/${dataset}/test_subsampled.jsonl"
            else
                data_file="${DATA_BASE}/${dataset}/test_subsampled_${var}.jsonl"
            fi

            if [ ! -f "$data_file" ]; then
                echo "⚠️  Data file not found: $data_file"
                continue
            fi

            echo "[$(date +%H:%M:%S)] Running: $dataset / $var / $model_name"

            if python run_searcho1_denseretriever.py \
                --dataset_name "$ds_arg" \
                --split test \
                --data_path "$data_file" \
                --fewshot 6 \
                --sample 1000 \
                --model_path "$model_path" \
                --retriever Contriever \
                --contriever_faiss_index_path "$CONTRIEVER_INDEX" \
                --contriever_meta_file_path "$CONTRIEVER_META" \
                --contriever_max_query_length 256; then
                echo "  ✅ Success"
            else
                echo "  ❌ Failed"
            fi
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
echo "📁 Results saved to: ./outputs/runs.contriever/"
echo ""
echo "Summary:"
echo "  - Datasets: ${#DATASETS[@]}"
echo "  - Variations: ${#VARIATIONS[@]}"
echo "  - Models: ${#MODELS[@]}"
echo "  - Total runs: $((${#DATASETS[@]} * ${#VARIATIONS[@]} * ${#MODELS[@]}))"
echo "============================================"
