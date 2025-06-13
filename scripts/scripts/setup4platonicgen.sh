#!/bin/bash
pip install --upgrade pip wheel setuptools requests && rm -rf ~/.cache/pip/
pip install -U xformers torch torchvision lightning diffusers transformers accelerate peft 
pip install hydra-core tyro imageio imageio-ffmpeg omegaconf ipdb plyfile lmdb roma lpips timm einops colorama wandb==0.17.7
pip install oss2 pcache-fileio pip wheel setuptools requests -i https://pypi.antfin-inc.com/simple
pip install --upgrade pyparsing pandas numpy opencv_python_headless
# For LPIPS
ossutil64 cp  oss://antsys-vilab/yyb/.cache/torch/hub/checkpoints/vgg16-397923af.pth /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth
rm -rf ~/.cache/pip/
rpm --rebuilddb && sudo yum install -y  glm-devel

ossutil64 cp -r  oss://antsys-vilab/yyb/PlatonicGen/third_party/diff-gaussian-rasterization/ ./diff-gaussian-rasterization/
cd ./diff-gaussian-rasterization && pip install . -vvv && cd ..


