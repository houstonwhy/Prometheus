#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/data1/yyb/PlatonicGen/
OUTPUT_DIR=./workdir/paper/stage1/

# # For teaser
for method_name in full; do
    for tasks in tartianair4view_easyvideo tartianair4view_hardvideo tartianair4view_mediumvideo; do
        python paper/stage1_eval.py --tasks=${tasks} --method_name=${method_name} --out_dir=${OUTPUT_DIR} 
        done
done

# for method_name in full full_randombg; do
#     for tasks in re10k; do
#         python paper/stage1_eval.py --tasks=${tasks} --method_name=${method_name} --out_dir=${OUTPUT_DIR} 
#         done
# done
# # legend -> hard
# for method_name in nodepth full_randombg full nodsv; do
#     for tasks in tartianair4view_legend; do
#         python paper/stage1_eval.py --tasks=${tasks} --method_name=${method_name} --out_dir=${OUTPUT_DIR} 
#         done
# done
# nightmare -> mdeium
# for method_name in nodepth full_randombg full nodsv; do
#     for tasks in tartianair4view_nightmare; do
#         python paper/stage1_eval.py --tasks=${tasks} --method_name=${method_name} --out_dir=${OUTPUT_DIR} 
#         done
# done

# hard -> mdeium
# for method_name in nodepth full_randombg full nodsv; do
#     for tasks in tartianair4view,tartianair4view_hard; do
#         python paper/stage1_eval.py --tasks=${tasks} --method_name=${method_name} --out_dir=${OUTPUT_DIR} 
#         done
# done

