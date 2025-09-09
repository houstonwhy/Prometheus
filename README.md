# Prometheus: 3D-Aware Latent Diffusion Models for Feed-Forward Text-to-3D Scene Generation (CVPR 2025)

<!-- 
[Jiahao Shao*](https://jhaoshao.github.io/), Yuanbo Yang*, Hongyu Zhou, [Youmin Zhang](https://youmi-zym.github.io/),  [Yujun Shen](https://shenyujun.github.io/), [Vitor Guizilini](https://vitorguizilini.github.io/), [Yue Wang](https://yuewang.xyz/), [Matteo Poggi](https://mattpoggi.github.io/), [Yiyi Liao](https://yiyiliao.github.io/ ) -->

[![Website](https://img.shields.io/badge/🔥website-Prometheus-orange)](https://freemty.github.io/project-prometheus/) [![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/pdf/2412.21117)

 <!-- [![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/jhshao/ChronoDepth)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/jhshao/ChronoDepth-v1) -->

> **Prometheus: 3D-Aware Latent Diffusion Models for Feed-Forward Text-to-3D Scene Generation** <br>

> Yuanbo Yang, Jiahao Shao and Xinyang, Li and Yujun, Shen and Andreas, Geiger and Yiyi, Liao <br>

## Abstract

<img src="./docs/assets/teaser_p1.jpg"/>

**Overview:** *We present a novel method for feed-forward scene-level 3D generation. At its core, our approach harnesses the power of 2D priors to fuel generalizable and efficient 3D synthesis – hence our name, <font color=#ff455c>Prometheus </font>🔥*


In this work, we introduce  <font color=#ff455c>Prometheus </font>🔥, a 3D-aware latent diffusion model for text-to-3D generation at both object and scene levels in seconds. We formulate 3D scene generation as multi-view, feed-forward, pixel-aligned 3D Gaussian generation within the latent diffusion paradigm. To ensure generalizability, we build our model upon pre-trained text-to-image generation model with only minimal adjustments, and further train it using a large number of images from both single-view and multi-view datasets. Furthermore, we introduce an RGB-D latent space into 3D Gaussian generation to disentangle appearance and geometry information, enabling efficient feed-forward generation of 3D Gaussians with better fidelity and geometry. Extensive experimental results demonstrate the effectiveness of our method in both feed-forward 3D Gaussian reconstruction and text-to-3D generation.

## Method

<img src="./docs/assets/method.jpg"/>

 Our training process is divided into two stages. In stage 1, our objective is to train a **GS-VAE**. Utilizing multi-view images along with their corresponding pseudo depth maps and camera poses, our GS-VAE is designed to encode these multi-view RGB-D images, integrate cross-view information, and ultimately decode them into pixel-aligned 3DGS. In stage 2, we focus on training a **MV-LDM**. We can generate multi-view RGB-D latents by sampling from randomly-sampled noise with trained MV-LDM.



## ⚙️ Installtion

All codes are succefully tested on:

- NVIDIA RTX-A6000 GPU (48G)
- Ubuntu 22.04.5 LTS
- CUDA 12.1
- Python 3.10
- Pytorch 2.4.0

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
    # Download the pretrained raydiff as the initialization of GSDecoder (Optional) or set in RayDiff: https://github.com/jasonyzhang/RayDiffusion
    gdown https://drive.google.com/uc\?id\=1anIKsm66zmDiFuo8Nmm1HupcitM6NY7e
    unzip models.zip -d ${PRETRAINED_PATH}/
    ```

6. Download pretrained Prometheus Checkpoint from [HuggingFace](https://huggingface.co/sumyyyyy/Prometheus_ckpt) to `./pretrained/full.ckpt`

## 🚀 Inference

Inference from prompt file

```bash
python inference.py --ckpt-path="pretrained/full.ckpt" \
                --prompts="benchmarks/benchmarks/gpt4v_gallery/prompt.txt" \ 
                --seed=42 \
                --out_dir="outputs/inference_results"
```

Or from text prompt input

```bash
python inference.py --ckpt-path="pretrained/full.ckpt" \
                --prompts="A large, abandoned factory repurposed as an urban exploration site, with large, empty spaces and rusting machinery." \
                --seed=42 \
                --device="cuda:0" \
                --out_dir="outputs/inference_results"
```

Eval on our Text-to-3D scene generation benchmark

```bash
python eval.py --ckpt-path="pretrained/full.ckpt" \
                --dataset-dir="benchmarks" \
                --tasks="scene_benchmark80" \
                --seed=42 \
                --devive="cuda:0" \
                --out-dir="ooutputs/eval_results"
```

## 🚗 Training (Doing)

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


## 📖 Related Projects

- [Director3D](https://github.com/imlixinyang/Director3D)
- [DepthAnything](https://github.com/DepthAnything/Depth-Anything-V2)
- [Camera as Rays](https://github.com/jasonyzhang/RayDiffusion)
- [gsplat](https://github.com/nerfstudio-project/gsplat)


## 🎓 Citation

Please cite our paper if you find this repository useful:

```bibtex
@article{yang2024prometheus,
      title={Prometheus: 3D-Aware Latent Diffusion Models for Feed-Forward Text-to-3D Scene Generation}, 
      author={Yuanbo, Yang and Jiahao, Shao and Xinyang, Li and Yujun, Shen and Andreas, Geiger and Yiyi, Liao},
      year={2024},
      journal= {arxiv:2412.21117},
}
```
