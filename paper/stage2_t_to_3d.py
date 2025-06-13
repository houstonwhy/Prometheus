"""Infernece code of Stage2's Eval"""
# pylint: disable=import-error
import os
import io
import base64
import tqdm
import pickle as pkl
from pathlib import Path
import random
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
import lightning
import numpy as np
import tyro
import torch
# from prometheus.utils.camera import sample_from_dense_cameras
from prometheus.utils.image_utils import colorize_depth_maps, postprocess_image
from prometheus.utils.visualization import plotly_scene_visualization
from einops import rearrange
import imageio 
# from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib.pyplot as plt


def merge_cfg(cfg, new_cfg):
    with open_dict(cfg):
        for key in new_cfg.keys():
            OmegaConf.update(cfg, key, new_cfg[key])
    return cfg

def concatenate_images(images, axis='width'):
    if axis == 'width':
        concatenated_images = rearrange(images, 'n h w c-> h (n w) c')
    elif axis == 'height':
        concatenated_images = rearrange(images, 'n h w c -> (n h) w c')
    else:
        raise ValueError("axis must be 'width' or 'height'")

    return concatenated_images

def view_color_coded_images(images, num_rows = -1, num_cols = -1, cmap  = "Spectral", c_range = [0,1]):
    if isinstance(images, list):
        num_frames = len(images)
    else:
        num_frames = images.shape[0]
    
    if num_rows == -1:
        num_rows = 1
        num_cols = num_frames

    figsize = (num_cols * 2, num_rows * 2)
    cmap = plt.get_cmap("Spectral")
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()
    for i in range(num_rows * num_cols):
        if i < num_frames:
            axs[i].imshow(images[i])
            for s in ["bottom", "top", "left", "right"]:
                axs[i].spines[s].set_color(cmap((c_range[1] - c_range[0]) * i / (num_frames) + c_range[0]))
                axs[i].spines[s].set_linewidth(5)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        else:
            axs[i].axis("off")
    # plt.tight_layout()

    return fig

_FULL_PATH = 'outputs/ckpts/prometheus_k8s_mvldm_dir3d_exp_mvldm_dataset_mvldm_viewcond_mvldm_dir3d_exp_exp12d_emanormfull/epoch=82-step=83000.ckpt'
MODELS_ = {
    'full' : _FULL_PATH,
    'final' : _FULL_PATH,
    'final_12d' : _FULL_PATH,
    # Ablation Training
    'no_sv' : 'outputs/ckpts/prometheus_k8s_mvldm_viewcond_exp_mvldm_dataset_mvldm_viewcond_mvldm_viewcond_exp_exp11a_mvldm_highnoise/epoch=12-step=13000.ckpt',
    'low_noise' : 'outputs/ckpts/prometheus_k8s_mvldm_mvrgbdt2i_exp_mvldm_dataset_mvldm_mvldm_mvrgbdt2i_exp_exp08c_mvldm_edm/epoch=157-step=158001.ckpt',
    'legacy_full' : 'outputs/ckpts/prometheus_k8s_mvldm_viewcond_exp_mvldm_dataset_mvldm_viewcond_mvldm_viewcond_exp_exp11a_mvldm_highnoise/epoch=12-step=13000.ckpt',
    'legacy_full_nohigh' : 'outputs/ckpts/prometheus_k8s_mvldm_mvrgbdt2i_exp_mvldm_dataset_mvldm_mvldm_mvrgbdt2i_exp_exp08c_mvldm_edm/epoch=99-step=100001.ckpt',
    # Ablation Inference
    'no_depth' : _FULL_PATH,
    'text_cfg': _FULL_PATH,
    'rescale0': _FULL_PATH,
}



@dataclass
class Args:
    method_name: str = 'legacy_full_nohigh'
    #text_file: str = 'sam1b'
    export_all: bool = True
    export_video: bool = True
    export_camera: bool = True
    export_ply: bool = True
    export_image: bool = True
    cfg_scale = 7.5
    cfg_type = 'hybrid'
    cfg_rescale = 0.7
    use_ema: bool = False
    inference_steps: int = 100
    render_size: int = 512
    num_input_views: int = 8
    # refiner SDS++
    num_refine_steps: int = 0
    max_scene_num: int = 100
    dataset_dir: str = 'benchmark'
    # tasks: str = 'dir3d_gallery,Re10K,DL3DV,MVImgNet,ACID'
    tasks: str = 'scene_benckmark'
    # MVImgNet,Re10K,DL3DV,Objaverse,ACID'
    # ACID,
    # dataset_dir: str = '/data1/yyb/PlatonicGen/benchmark/t3bench/'
    # tasks: str = 'prompt_single,prompt_surr'
    out_dir: str = '/data1/yyb/prometheus_baselines/'
    num_samples: int = 1
    gpu: int = 0
    seed: int = 42
    extra_filename: str = ''

def run_inference(args: Args):
    # print(args)
    device = f'cuda:{args.gpu}'
    args.export_video = args.export_video or args.export_all
    args.export_camera = args.export_camera or args.export_all
    args.export_ply = args.export_ply or args.export_all
    args.export_image = args.export_image or args.export_all
    torch.backends.cudnn.benchmark = True
    lightning.seed_everything(args.seed)
    num_input_views = args.num_input_views
    render_size = args.render_size
    num_samples = args.num_samples
    cfg_type, cfg_scale = args.cfg_type, args.cfg_scale
    cfg_rescale = 0.7


    tasks = args.tasks.split(',')
    dataset_dir = Path(args.dataset_dir)
    method_name = args.method_name
    ckpt_path =  MODELS_[method_name]
    
    from prometheus.utils.io import export_ply_for_gaussians, import_str, export_video
    # load additonal GS decoder
    decoder_ckpt = 'outputs/ckpts/prometheus_k8s_gsdecoder_exp_gsdecoder_dataset_full_gsdecoder_dit_gsdecoder_exp_exp06h_objaverse/epoch=215-step=215999.ckpt'
    if method_name in ['legacy_full', 'no_sv', 'legacy_full_nohigh', 'low_noise']:
        if False:
            params_decoder = torch.load(decoder_ckpt, weights_only=False, map_location='cpu')
        else:
            with open(decoder_ckpt, 'rb') as f:
                os.fsync(f.fileno()) # avoid stuck in loading
                buffer = io.BytesIO(f.read())
            params_decoder = torch.load(buffer, weights_only=False, map_location='cpu')

        cfg_decoder = params_decoder['hyper_parameters']
        new_cfg = OmegaConf.load('configurations/global_env/local_workstation.yaml')
        cfg_decoder = merge_cfg(cfg_decoder, new_cfg)
        decoder_system = import_str(cfg_decoder['experiment']['training']['module'])(opt = cfg_decoder, mode = 'inference').eval().to(device).to(torch.bfloat16)
        decoder_system.load_state_dict(params_decoder['state_dict'], strict = False)
        # del decoder_system
        gsdecoder_ext = decoder_system.model
    else:
        gsdecoder_ext = None
    
    # load MVLDM
    if True:
        params = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    else:
        with open(ckpt_path, 'rb') as f:
            os.fsync(f.fileno()) # avoid stuck in loading
            buffer = io.BytesIO(f.read())
        params = torch.load(buffer, weights_only=False, map_location='cpu')
    cfg = params['hyper_parameters']
    
    use_ema = args.use_ema

    new_cfg = OmegaConf.load('configurations/global_env/local_workstation.yaml')
    cfg = merge_cfg(cfg, new_cfg)
    system = import_str(cfg['experiment']['training']['module'])(opt = cfg, mode = 'inference').eval().to(device).to(torch.bfloat16)
    if use_ema:
        ema_state_dict = {}
        for k, v in params['state_dict'].items():
            if 'model_ema' in k:
                ema_state_dict[k.replace('model_ema','model')] = v
            elif ('disp' in k) or ('gsdecoder' in k):
                ema_state_dict[k] = v
        system.load_state_dict(ema_state_dict, strict = False)
    else:
        try:
            system.load_state_dict(params['state_dict'], strict = False)
        except:
            state_dict_clean = params['state_dict']
            keys_to_delete = [key for key, param in state_dict_clean.items() 
            if 'decoder' in key]
            for key in keys_to_delete:
                del state_dict_clean[key]
            system.load_state_dict(state_dict_clean, strict = False)
    if gsdecoder_ext is not None:
        system.gsdecoder = gsdecoder_ext
    # Re config schdeuler for legacy ckpts -> set p_data=0.5

        system.high_noise_level = True
        system.configure_noise_scheduler(1.)
    
    if method_name == 'text_cfg':
        cfg_type, cfg_scale = 'text', 7.5
    elif method_name == 'nodepth':
        cfg_type, cfg_scale = 'joint', 3.5
    elif method_name == 'legacy_full_nohigh':
        cfg_type, cfg_scale = 'joint', 7.5
        cfg_rescale = 0.7

    for task_name in tasks:
        # try:
        scene_list_dir = dataset_dir / task_name  / 'camera'
        out_dir = Path(args.out_dir) / method_name / task_name
        scene_list = os.listdir(scene_list_dir)
        scene_list = [os.path.join(scene_list_dir, file) for file in scene_list if file.endswith('pkl')]
        random.shuffle(scene_list)
        scene_list = scene_list[:args.max_scene_num]
        # scene_list.sort()
        # except:
        #     continue
        with torch.no_grad():
            for _, scene_path in enumerate(scene_list):
                # if str(out_dir / filename / f"{i}" / filename)

                # for idx in range(len(dataset)):
                with open(scene_path, 'rb') as fp:
                    data = pkl.load(fp)
                cameras, text = torch.tensor(data['cameras']).to(device), data['text']
                # text = 'Cyberpunk style,' + text
                if len(cameras.shape) == 2:
                    cameras = cameras[None]

                for i in range(num_samples):
                    # filename = text[:80].replace(' ', '_')
                    sparse_cameras = cameras[i:i+1, ::int((cameras.shape[1]-1)/(num_input_views-1))]

                    filename = text[:120].replace(' ', '_')
                    save_dir = out_dir / filename / f"{i}"
                    os.makedirs(save_dir, exist_ok=True)
                    if os.path.exists(save_dir / f'{filename}.mp4'):
                        continue

                    cam_vis = plotly_scene_visualization(cameras[i], img_return=True, key_frame_rate=int((cameras.shape[1]-1)/(num_input_views-1)), dist=-1.2)
                    cam_vis.save(save_dir / f'cam_{filename}.png')
                
                    with torch.amp.autocast("cuda", enabled=True):
                        result = system.inference(
                            cameras = sparse_cameras, 
                            num_inference_steps=args.inference_steps, 
                            dense_cameras = cameras[i:i+1],
                            guidance_type = cfg_type,
                            guidance_scale=cfg_scale,
                            cfg_rescale = cfg_rescale,
                            text = [text],
                            num_input_views = num_input_views,
                            render_size=render_size,
                            gs_decoder_ext=gsdecoder_ext,
                            # refiner=refiner if num_refine_steps > 0 else None
                            )

                    if args.export_image:
                    # save raw rgb
                        depths_pred = postprocess_image(colorize_depth_maps(result['depths_pred'], None, None,"Spectral_r"), 0, 1)
                        images_pred = postprocess_image(result['images_pred'][0], -1, 1)
                        depths_gs_render = postprocess_image(colorize_depth_maps(result['depths_gs_render'], None, None,"Spectral"), 0, 1)
                        images_gs_render = postprocess_image(result['images_gs_render'][0], -1, 1)
                        # visualize pose
                        imageio.imsave(str(save_dir / 'image.png'),
                                        np.concatenate((concatenate_images(depths_pred),concatenate_images(images_pred)), axis = 0))
                        imageio.imsave(str(save_dir / 'image_gs.png'),
                                        np.concatenate((concatenate_images(depths_gs_render),concatenate_images(images_gs_render)), axis = 0))
                        
                        full_dir = save_dir / 'full'
                        os.makedirs(full_dir, exist_ok=True)
                        for j in range(8):
                            imageio.imsave(str(full_dir / f'images_pred_{j}.png'),images_pred[j])
                            imageio.imsave(str(full_dir / f'depths_pred_{j}.png'),depths_pred[j])
                            imageio.imsave(str(full_dir / f'depths_gs_render_{j}.png'),depths_gs_render[j])
                            imageio.imsave(str(full_dir / f'images_gs_render_{j}.png'),images_gs_render[j])
                    
                    if args.export_ply:
                        # os.makedirs(os.path.join(out_dir, 'ply'), exist_ok=True)
                        export_ply_for_gaussians(os.path.join(save_dir, f'{filename}'), result['gaussians'])

                    if args.export_video:
                        if hasattr(system, 'gsdecoder'):
                            gsdecoder = system.gsdecoder
                        else:
                            gsdecoder = gsdecoder_ext
                        video_frame_dir = str(out_dir / filename / f"{i}" / filename)
                        os.makedirs(video_frame_dir,exist_ok=True)
                        render_fn = lambda cameras, h=render_size, w=render_size: gsdecoder.render(cameras, result['gaussians'], h=h, w=w, bg_color=(1,1,1))[:2]
                        full_images, full_depths = export_video(render_fn, save_dir , filename, cameras[i:i+1], device=device, fps=30, num_frames=120, render_size=render_size)
                        for j, pp in enumerate(full_images):
                                imageio.imsave(os.path.join(video_frame_dir, str(j).zfill(6) + ".png"), pp)
                

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_inference(args)


