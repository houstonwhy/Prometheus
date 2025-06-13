#!/bin/bash
set -x -e

input_dirs=(
    # "/home/jiahao/workspace/LGM/outputs/lgm"
    # "/home/jiahao/workspace/LGM/outputs/director3d"
    # "/data0/jhshao/prometheus_baselines/gaussiandreamer"
    # "/data1/yyb/prometheus_baselines/low_noise"
    "/data1/yyb/prometheus_baselines/textcfg"
    "/data1/yyb/prometheus_baselines/cfgrescale0"
)

# sub_dirs=("prompt_single" "prompt_multi" "prompt_surr")
sub_dirs=("prompt_surr")

for input_dir in "${input_dirs[@]}"; do
    for sub_dir in "${sub_dirs[@]}"; do
        python compute_brisque.py --input_dir "${input_dir}/${sub_dir}"
    done
done