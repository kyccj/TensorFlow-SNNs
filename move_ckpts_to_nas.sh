#!/bin/bash
# Move checkpoint files to NAS, preserving directory structure
# Skips currently running experiments (_adaptive_lambda_v2/)
#
# Source: /home/kyccj/PycharmProjects/TensorFlow-SNNs/<exp_dir>/.../*.weights.h5
# Dest:   /media/hdd1/kyccj/EIP/<exp_dir>/.../*.weights.h5

SRC="/home/kyccj/PycharmProjects/TensorFlow-SNNs"
DST="/media/hdd1/kyccj/EIP"
SKIP="_adaptive_lambda_v2"

echo "=== Checkpoint Migration to NAS ==="
echo "Source: $SRC"
echo "Dest:   $DST"
echo "Skip:   $SKIP (running experiments)"
echo ""

# Count and size before
total_files=0
total_size=0

while IFS= read -r f; do
    # Skip running experiments
    rel="${f#$SRC/}"
    if [[ "$rel" == ${SKIP}/* ]]; then
        continue
    fi

    size=$(stat --format=%s "$f" 2>/dev/null)
    total_files=$((total_files + 1))
    total_size=$((total_size + size))
done < <(find "$SRC" -name "*.weights.h5" -type f 2>/dev/null)

echo "Files to move: $total_files"
echo "Total size: $((total_size / 1024 / 1024 / 1024))GB ($((total_size / 1024 / 1024))MB)"
echo ""

# Move files
moved=0
failed=0

while IFS= read -r f; do
    rel="${f#$SRC/}"

    # Skip running experiments
    if [[ "$rel" == ${SKIP}/* ]]; then
        continue
    fi

    dst_file="$DST/$rel"
    dst_dir=$(dirname "$dst_file")

    # Create directory
    mkdir -p "$dst_dir"

    # Move file
    if mv "$f" "$dst_file" 2>/dev/null; then
        moved=$((moved + 1))
        # Get experiment dir name for display
        exp_dir=$(echo "$rel" | cut -d'/' -f1)
        ckpt_name=$(basename "$f")
        echo "[OK] $exp_dir/$ckpt_name"
    else
        failed=$((failed + 1))
        echo "[FAIL] $rel"
    fi
done < <(find "$SRC" -name "*.weights.h5" -type f 2>/dev/null | sort)

echo ""
echo "=== Done ==="
echo "Moved: $moved files"
echo "Failed: $failed files"
echo "Freed: ~$((total_size / 1024 / 1024 / 1024))GB"

# Verify
echo ""
echo "=== NAS contents ==="
du -sh "$DST"/* 2>/dev/null | sort -rh | head -20
