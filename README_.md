# Prometheus

## Installation

1. Create a new conda environment:
    ```bash
    conda create -n prometheus python==3.10
    conda activate prometheus
    ```

2. Install PyTorch 2.4.0 (or use your own if it is compatible with xformers) under CUDA 12.1:
    ```bash
    pip install torch==2.4.0 torchvision==0.19.0 lightning==2.4.0 diffusers==0.30.0 transformers==4.44.1 xformers==0.0.27.post2
    ```

3. Install other packages:
    ```bash
    pip install hydra-core tyro imageio imageio-ffmpeg omegaconf ipdb plyfile lmdb roma lpips timm einops colorama wandb peft opencv-python 
    ```
    Install gsplat from pre-compiled wheel:
    ```bash
    pip install https://github.com/nerfstudio-project/gsplat/releases/download/v1.4.0/gsplat-1.4.0%2Bpt24cu121-cp310-cp310-linux_x86_64.whl
    ```

4. Set up the output directory and Hugging Face directory (if you need to download HF checkpoints) in `configurations/config.yaml`. Additionally, you can set the dataset root in `configurations/global_env` if you need to run on different environments (e.g., debug on your own device and train on a cluster):
    ```yaml
    output_dir: ./YOUR_OUTPUT_DIR
    PRETRAINED_PATH: ./PRETRAINED_PATH
    HF_PATH: ./YOUR_HF_PATH
    wandb:
      entity: xxxx-university # wandb account name / organization name [fixme]
      mode: online # set wandb logging to online, offline, or dryrun
    ```

5. Download pretrained models (VGG16, DepthAnying-V2, SD-21):
    Huggingface ckpts
    ```bash
    HF_PATH=$YOUR_HF_PATH
    HF_MIRROR='https://huggingface.co'
    # hf_endpoint='https://hf-mirror.com' or use hf mirror for people under GFW 
    python tools/download_hf_model.py --repo_name='depth-anything/Depth-Anything-V2-Small-hf' --local_dir=${HF_PATH} --endpoint=${HF_MIRROR}
    python tools/download_hf_model.py --repo_name='stabilityai/stable-diffusion-2-1' --filename='v2-1_768-ema-pruned.ckpt' --local_dir=${HF_PATH} --endpoint=${HF_MIRROR}
    ```

    
    ```bash
    PRETRAINED_PATH=./pretrained/
    # Download the pretrained raydiff as the initialization of GSDecoder (Optional) or set in 
    RayDiff: https://github.com/jasonyzhang/RayDiffusion
    gdown https://drive.google.com/uc\?id\=1anIKsm66zmDiFuo8Nmm1HupcitM6NY7e
    unzip models.zip ${PRETRAINED_PATH}/
    ```

5. Inference from text prompt
    A training script example, you can set available GPUs via `GPUS=[0,1,2]` or in `configurations/global_env/base_cluster.yaml`
    - Training script of Stage 1 (GSDecoder) 
    ```bash
    python train.py tags='exp06h_gsdecoder_objaverse' global_env=aistudio_k8s dataset=gsdecoder_dataset_full algorithm=gsdecoder_dit experiment=gsdecoder_exp experiment.training.batch_size=2 experiment.training.accumulate_grad_batches=2
    ```


5. Eval on our scene generation benchmatk

    ```bash
    python train.py tags='exp06h_gsdecoder_objaverse' global_env=aistudio_k8s dataset=gsdecoder_dataset_full algorithm=gsdecoder_dit experiment=gsdecoder_exp experiment.training.batch_size=2 experiment.training.accumulate_grad_batches=2
    ```

6. Start training
    A training script example, you can set available GPUs via `GPUS=[0,1,2]` or in `configurations/global_env/base_cluster.yaml`
    - Training script of Stage 1 (GSDecoder) 
    ```bash
    python train.py tags='exp06h_gsdecoder_objaverse' global_env=aistudio_k8s dataset=gsdecoder_dataset_full algorithm=gsdecoder_dit experiment=gsdecoder_exp experiment.training.batch_size=2 experiment.training.accumulate_grad_batches=2
    ```
    - Training script of Stage 2 (MVLDM)
    ```bash
    python train.py experiment=mvldm_vpred_exp algorithm=mvldm_viewcond dataset=mvldm_dataset experiment.image_size=256 experiment.training.batch_size=4 experiment.validation.batch_size=4 experiment.training.single_view_num=4 experiment.training.use_gsdecoder=true experiment.training.mvldm_path='' tags='exp12a_mvldm_vpredfull3d' global_env=aistudio_k8s experiment.training.accumulate_grad_batches=1
    ```
    Joint tuning of the full model (Optional)
    ```bash
    python train.py experiment=mvldm_dir3d_exp algorithm=mvldm_viewcond dataset=mvldm_dataset experiment.image_size=256 experiment.training.batch_size=2 experiment.validation.batch_size=2 experiment.training.rendering_batch_size=2 experiment.training.single_view_num=2 tags='exp12d_emanormfull' global_env=aistudio_k8s experiment.training.accumulate_grad_batches=1 experiment.training.learning_rate=2e-5 experiment.training.tune_decoder_only=false gsdecoder.network.use_ema_norm=true
    ```