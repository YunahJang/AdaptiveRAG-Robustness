#!/bin/bash
# Copy our modified/added baseline files over freshly cloned upstream repos.
#
#   export QR_ROOT=/path/to/QueryRobustness
#   mkdir -p $QR_ROOT/baselines && cd $QR_ROOT/baselines
#   git clone https://github.com/s-nlp/AdaRAGUE.git
#   git clone https://github.com/RUC-NLPIR/Search-o1.git
#   $QR_ROOT/baselines_overlay/apply_overlay.sh [target_dir]
#
# target_dir defaults to $QR_ROOT/baselines.
set -euo pipefail

OVERLAY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QR_ROOT="${QR_ROOT:-$(cd "$OVERLAY_DIR/.." && pwd)}"
TARGET_DIR="${1:-$QR_ROOT/baselines}"

for repo in AdaRAGUE Search-o1; do
    if [ ! -d "$TARGET_DIR/$repo" ]; then
        echo "error: $TARGET_DIR/$repo not found - clone the baseline repos first:" >&2
        echo "  mkdir -p $TARGET_DIR && cd $TARGET_DIR" >&2
        echo "  git clone https://github.com/s-nlp/AdaRAGUE.git" >&2
        echo "  git clone https://github.com/RUC-NLPIR/Search-o1.git" >&2
        exit 1
    fi
done

if [ "$(cd "$TARGET_DIR" && pwd)" = "$OVERLAY_DIR" ]; then
    echo "error: target directory must differ from the overlay directory." >&2
    exit 1
fi

copied=0
while IFS= read -r -d '' src; do
    rel="${src#"$OVERLAY_DIR"/}"
    case "$rel" in
        README.md | apply_overlay.sh) continue ;;
    esac
    dst="$TARGET_DIR/$rel"
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    echo "  $rel"
    copied=$((copied + 1))
done < <(find "$OVERLAY_DIR" -type f -print0)

echo "copied $copied files into $TARGET_DIR"
