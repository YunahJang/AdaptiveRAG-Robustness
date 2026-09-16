#!/bin/bash
# Batch execution script for Contriever experiments
# Usage: ./run_batch.sh [dataset] [method]
# Example: ./run_batch.sh hotpotqa DRAGIN
# Example: ./run_batch.sh all all  (runs everything)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"

# Parse arguments
DATASET=${1:-all}
METHOD=${2:-all}

# Available options
DATASETS=("natural_questions" "triviaqa" "squad" "hotpotqa" "2wikimultihopqa" "musique")
METHODS=("wo-RAG" "SR-RAG" "DRAGIN")
VARIATIONS=("original" "human" "formality" "readability" "declarative" "imperative" "grammar_error" "spelling_error" "keyboard_error" "ocr_error")

echo "============================================"
echo "Contriever Batch Execution"
echo "============================================"
echo "Dataset: $DATASET"
echo "Method: $METHOD"
echo "============================================"
echo ""

# Determine which datasets to run
if [ "$DATASET" = "all" ]; then
    DATASETS_TO_RUN=("${DATASETS[@]}")
else
    DATASETS_TO_RUN=("$DATASET")
fi

# Determine which methods to run
if [ "$METHOD" = "all" ]; then
    METHODS_TO_RUN=("${METHODS[@]}")
else
    METHODS_TO_RUN=("$METHOD")
fi

# Run experiments
total=0
success=0
failed=0

for ds in "${DATASETS_TO_RUN[@]}"; do
    for mt in "${METHODS_TO_RUN[@]}"; do
        for var in "${VARIATIONS[@]}"; do
            config_file="$SCRIPT_DIR/$ds/${mt}_${var}.json"

            if [ ! -f "$config_file" ]; then
                echo "⚠️  Config not found: $config_file"
                continue
            fi

            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "🚀 Running: $ds / $mt / $var"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            cd "$SRC_DIR" || exit 1

            if python main_newstore.py -c "$config_file"; then
                echo "✅ Success: $ds / $mt / $var"
                ((success++))
            else
                echo "❌ Failed: $ds / $mt / $var"
                ((failed++))
            fi

            ((total++))
        done
    done
done

echo ""
echo "============================================"
echo "Batch Execution Complete"
echo "============================================"
echo "Total: $total"
echo "Success: $success"
echo "Failed: $failed"
echo "============================================"
