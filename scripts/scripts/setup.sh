
#Step1 Inatall pytorch family
ossutil64 cp oss://antsys-vilab/softwares/wheels/torch-1.10.2+cu113-cp38-cp38-linux_x86_64.whl ./ &&ossutil64 cp oss://antsys-vilab/softwares/wheels/torchvision-0.11.3+cu113-cp38-cp38-linux_x86_64.whl ./ && \
pip uninstall torch torchvision -y && \
pip install torch-1.10.2+cu113-cp38-cp38-linux_x86_64.whl torchvision-0.11.3+cu113-cp38-cp38-linux_x86_64.whl &&\
rm -rf torch-1.10.2+cu113-cp38-cp38-linux_x86_64.whl torchvision-0.11.3+cu113-cp38-cp38-linux_x86_64.whl

# #Step2 Requirements
pip install -i htps://pypi.antfin-inc.com/simple --upgrade --upgrade-strategy only-if-needed  -r requirements/dev.txt && pip install mmcv-full==1.4.5  mmdet==2.20.0 --upgrade --upgrade-strategy only-if-needed  # mmdet cannot run on 8 GPUs with 2.28.2

# #Step3 BEVFusion/Diffuser/
cd MagicDriveSVD/third_party/bevfusion && pip install -e . && \
cd ../diffusers && pip install -vvv . && \
cd ../xformers && pip install -vvv . && cd .. && cd .. && \
pip install deepspeed --upgrade --upgrade-strategy only-if-needed &&\
pip uninstall opencv-python -y && pip install opencv-python-headless==4.5.5.62

# #Step2 Ckeckpoints
# ossutil64 cp oss://antsys-vilab/checkpoints/huggingface/stable-diffusion-v1-5/ ./pretrained
# ossutil64 cp oss://antsys-vilab/yyb/pretrain/MagicDrive/SDv1.5mv-rawbox_2023-09-07_18-39_224x400.zip

#Bug -> ImportError: libGL.so.1: cannot open shared object file: No such file or directory


# Soft-link data and ckpt from nas
# rm -rf data && ln -s /input/yyb/magicdrivesvd/data/ ./ && rm -rf pretrained && ln -s /input/yyb/magicdrivesvd/pretrained/ ./