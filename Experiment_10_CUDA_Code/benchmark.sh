#!/usr/bin/env bash
set -e
for f in matrix_global matrix_tiled reduction prefix_sum histogram; do
  nvcc -O2 "${f}.cu" -o "$f"
done
echo "All Experiment 10 examples compiled."
