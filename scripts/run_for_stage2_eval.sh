#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/data1/yyb/PlatonicGen/
DATASET_ROOT=/data1/yyb/PlatonicGen/benchmark/t3bench
OUTPUT_DIR=/data1/yyb/prometheus_baselines/
for method_name in legacy_full; do
    for tasks in scene_benckmark; do
        python paper/stage2_t_to_3d.py --dataset_dir=${DATASET_ROOT} --tasks=${tasks} --method_name=${method_name}
        done
done