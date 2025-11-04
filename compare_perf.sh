#!/bin/bash
# Compare all implementations with perf

if [ $# -lt 1 ]; then
    echo "Usage: $0 <size> [mode]"
    echo "Example: $0 512 cache"
    exit 1
fi

SIZE=$1
MODE=${2:-stat}

echo "╔════════════════════════════════════════╗"
echo "║  Matrix Multiplication Performance     ║"
echo "║  Size: ${SIZE}×${SIZE}"
echo "╚════════════════════════════════════════╝"
echo ""

for algo in naive blocked simd nalgebra; do
    echo "┌────────────────────────────────────────┐"
    echo "│ Algorithm: $(printf '%-28s' $algo) │"
    echo "└────────────────────────────────────────┘"
    ./profile.sh $algo $SIZE $MODE 2>&1
    echo ""
    echo ""
done

echo "╔════════════════════════════════════════╗"
echo "║  Comparison complete!                  ║"
echo "╚════════════════════════════════════════╝"
