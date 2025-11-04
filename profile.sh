#!/bin/bash
# Perf profiling script for matrix multiplication implementations

BINARY="./target/x86_64-unknown-linux-gnu/release/profile_matmul"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <algorithm> <size> [perf-mode]"
    echo ""
    echo "Algorithms: naive, blocked, simd, fma, nalgebra"
    echo "Size: Matrix dimension (e.g., 256, 512, 1024)"
    echo "Perf modes:"
    echo "  stat     - Quick statistics (default)"
    echo "  cache    - Detailed cache analysis"
    echo "  record   - Record for later analysis with 'perf report'"
    echo "  all      - Comprehensive stats"
    exit 1
fi

ALGORITHM=$1
SIZE=$2
MODE=${3:-stat}

echo "========================================"
echo "Profiling: $ALGORITHM (${SIZE}×${SIZE})"
echo "========================================"
echo ""

case $MODE in
    stat)
        # Basic performance statistics
        perf stat -d \
            $BINARY $ALGORITHM $SIZE
        ;;

    cache)
        # Detailed cache analysis
        perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
            $BINARY $ALGORITHM $SIZE
        ;;

    record)
        # Record for detailed analysis
        perf record -g \
            $BINARY $ALGORITHM $SIZE
        echo ""
        echo "Profile recorded! Run 'perf report' to analyze."
        ;;

    all)
        # Comprehensive statistics
        perf stat -e cycles,instructions,cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,\
LLC-loads,LLC-load-misses,LLC-stores,\
branch-instructions,branch-misses \
            $BINARY $ALGORITHM $SIZE
        ;;

    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac
