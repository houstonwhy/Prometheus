
#Step1 Inatall pytorch family
ossutil64 cp oss://antsys-vilab/softwares/wheels/torch-1.10.2+cu113-cp38-cp38-linux_x86_64.whl ./
ossutil64 cp oss://antsys-vilab/softwares/wheels/torchvision-0.11.3+cu113-cp38-cp38-linux_x86_64.whl ./

pip install torch-1.10.2+cu113-cp38-cp38-linux_x86_64.whl torchvision-0.11.3+cu113-cp38-cp38-linux_x86_64.whl
# ossutil64 cp oss://antsys-vilab/softwares/wheels/torchvision-0.16.0+cu118-cp38-cp38-linux_x86_64.whl .
# ossutil64 cp oss://antsys-vilab/softwares/wheels/xformers-0.0.22.post4+cu118-cp38-cp38-manylinux2014_x86_64.whl .
# ossutil64 cp  oss://antsys-vilab/softwares/wheels/torch-2.1.0+cu118-cp38-cp38-linux_x86_64.whl .

rm -rf install torch-1.10.2+cu113-cp38-cp38-linux_x86_64.whl torchvision-0.11.3+cu113-cp38-cp38-linux_x86_64.whl

#Step2 Requirements
pip install -i htps://pypi.antfin-inc.com/simple --upgrade --upgrade-strategy only-if-needed  -r requirements/dev.txt
pip install mmcv-full==1.4.5  mmdet==2.20.0 --upgrade --upgrade-strategy only-if-needed  # mmdet cannot run on 8 GPUs with 2.28.2

#Step3 Requirements
cd third_party
cd bevfusion

#Step2 Ckeckpoints
ossutil64 cp oss://antsys-vilab/checkpoints/huggingface/stable-diffusion-v1-5/ ./pretrained
ossutil64 cp oss://antsys-vilab/yyb/pretrain/MagicDrive/SDv1.5mv-rawbox_2023-09-07_18-39_224x400.zip


ossutil64 cp oss://antsys-vilab/softwares/wheels/torch-2.1.0+cu118-cp38-cp38-linux_x86_64.whl .
ossutil64 cp oss://antsys-vilab/softwares/wheels/torchvision-0.16.0+cu118-cp38-cp38-linux_x86_64.whl .
ossutil64 cp oss://antsys-vilab/softwares/wheels/xformers-0.0.22.post4+cu118-cp38-cp38-manylinux2014_x86_64.whl .

# Torch 2.1 version
ossutil64 cp oss://antsys-vilab/softwares/wheels/torch-2.1.0+cu118-cp38-cp38-linux_x86_64.whl .
ossutil64 cp oss://antsys-vilab/softwares/wheels/torchvision-0.16.0+cu118-cp38-cp38-linux_x86_64.whl .
ossutil64 cp oss://antsys-vilab/softwares/wheels/xformers-0.0.22.post4+cu118-cp38-cp38-manylinux2014_x86_64.whl .
pip uninstall torch torchvison
pip install torch-2.1.0+cu118-cp38-cp38-linux_x86_64.whl torchvision-0.16.0+cu118-cp38-cp38-linux_x86_64.whl xformers-0.0.22.post4+cu118-cp38-cp38-manylinux2014_x86_64.whl  
# rm -rf torch-2.1.0+cu118-cp38-cp38-linux_x86_64.whl torchvision-0.16.0+cu118-cp38-cp38-linux_x86_64.whl xformers-0.0.22.post4+cu118-cp38-cp38-manylinux2014_x86_64.whl
# update diffuser
pip install diffusers==0.26 --upgrade --upgrade-strategy only-if-needed


# Torch 1.13 version
ossutil64 cp oss://antsys-vilab/softwares/wheels/torch-1.13.1+cu116-cp38-cp38-linux_x86_64.whl .
ossutil64 cp oss://antsys-vilab/softwares/wheels/torchvision-0.14.1+cu116-cp38-cp38-linux_x86_64.whl .
pip install torchvision-0.14.1+cu116-cp38-cp38-linux_x86_64.whl torch-1.13.1+cu116-cp38-cp38-linux_x86_64.whl
pip install diffusers==0.26 xformers==0. --upgrade --upgrade-strategy only-if-needed