#!/usr/bin/env bash
set -e
nvcc -O2 -std=c++14 cpu_basic_tiled.cu -lcublas -o experiment11
echo "Built ./experiment11"
echo "Run with: ./experiment11"
