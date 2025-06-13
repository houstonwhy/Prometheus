#!/bin/bash
# ossutil64 cp -r  oss://antsys-vilab/yyb/PlatonicGen/third_party/diff-gaussian-rasterization/ ./diff-gaussian-rasterization/
# cd ./diff-gaussian-rasterization && pip install . -vvv && cd ..
ossutil64 cp oss://antsys-vilab/yyb/wheels/gsplat-1.4.0+pt24cu121-cp310-cp310-linux_x86_64.whl ./
pip install gsplat-1.4.0+pt24cu121-cp310-cp310-linux_x86_64.whl
pip install --upgrade pyparsing easydict