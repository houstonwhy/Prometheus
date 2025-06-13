"""Infernece code of Stage2's Eval Prometheus Benchmark"""
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
from einops import rearrange
import imageio 
from dataclasses import dataclass
from prometheus.utils.image_utils import colorize_depth_maps, postprocess_image, concatenate_images
from prometheus.utils.visualization import plotly_scene_visualization
from prometheus.utils.io import export_ply_for_gaussians, import_str, export_video



def merge_cfg(cfg, new_cfg):
    with open_dict(cfg):
        for key in new_cfg.keys():
            OmegaConf.update(cfg, key, new_cfg[key])
    return cfg


@dataclass
class Args:
    method_name: str = 'prometheus'
    ckpt_path: str = 'pretrained/full.ckpt'
    global_cfg_path: str = 'configurations/global_env/local_workstation.yaml'
    export_all: bool = True
    export_video: bool = True
    export_camera: bool = True
    export_ply: bool = True
    export_image: bool = True
    cfg_scale = 7.5
    cfg_type = 'joint'
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
    out_dir: str = 'outputs/eval_results'
    num_samples: int = 1
    device: str = 'cuda:0'
    seed: int = 42
    extra_filename: str = ''
def run_eval(args: Args):
    # print(args)
    device = args.device
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
    cfg_rescale = args.cfg_rescale
    tasks = args.tasks.split(',')
    dataset_dir = Path(args.dataset_dir)
    ckpt_path =  args.ckpt_path
    method_name = args.method_name


    params = torch.load(ckpt_path, weights_only=False, map_location='cpu')

    # load additonal GS decoder
    params_decoder = params['decoder']
    cfg_decoder = params_decoder['hyper_parameters']
    new_cfg = OmegaConf.load('configurations/global_env/local_workstation.yaml')
    cfg_decoder = merge_cfg(cfg_decoder, new_cfg)
    decoder_system = import_str(cfg_decoder['experiment']['training']['module'])(opt = cfg_decoder, mode = 'inference').eval().to(device).to(torch.bfloat16)
    decoder_system.load_state_dict(params_decoder['state_dict'], strict = False)
    gsdecoder_ext = decoder_system.model
    
    
    # load main model
    params_mvldm = params['mvldm']
    cfg = params_mvldm['hyper_parameters']
    new_cfg = OmegaConf.load('configurations/global_env/local_workstation.yaml')
    cfg = merge_cfg(cfg, new_cfg)
    system = import_str(cfg['experiment']['training']['module'])(opt = cfg, mode = 'inference').eval().to(device).to(torch.bfloat16)
    system.load_state_dict(params_mvldm['state_dict'], strict = False)
    system.gsdecoder = gsdecoder_ext


    for task_name in tasks:
        # try:
        scene_list_dir = dataset_dir / task_name  / 'camera'
        out_dir = Path(args.out_dir) / method_name / task_name
        scene_list = os.listdir(scene_list_dir)
        scene_list = [os.path.join(scene_list_dir, file) for file in scene_list if file.endswith('pkl')]
        random.shuffle(scene_list)
        scene_list = scene_list[:args.max_scene_num]

        with torch.no_grad():
            for _, scene_path in enumerate(scene_list):
                with open(scene_path, 'rb') as fp:
                    data = pkl.load(fp)
                cameras, text = torch.tensor(data['cameras']).to(device), data['text']
                if len(cameras.shape) == 2:
                    cameras = cameras[None]

                for i in range(num_samples):
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
    run_eval(args)


