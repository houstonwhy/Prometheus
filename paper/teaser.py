"""Infernece code Stage1 evalaution Sce4.1.1"""
# pylint: disable=import-error
import os
import io
import base64
import tqdm
import pickle as pkl
from pathlib import Path
import random
from omegaconf import OmegaConf
import imageio
from omegaconf.omegaconf import open_dict
import lightning
import numpy as np
import tyro
import torch
from prometheus.utils.io import export_ply_for_gaussians, import_str, export_video
# from prometheus.utils.camera import sample_from_dense_cameras
from prometheus.utils.image_utils import colorize_depth_maps, postprocess_image, concatenate_images
# from prometheus.utils.visualization import plotly_scene_visualization
# from einops import rearrange
# import imageio 
# from abc import ABC, abstractmethod
from dataclasses import dataclass

from prometheus.datasets import RealEstate10KDatasetEval

def merge_cfg(cfg, new_cfg):
    with open_dict(cfg):
        for key in new_cfg.keys():
            OmegaConf.update(cfg, key, new_cfg[key])
    return cfg

torch.backends.cudnn.benchmark = True

TASKS_ = {
        'tartianair2view' : (
            'tartianair',
            '/data0/jhshao/prometheus_benchmark/tartanair',
            '/home/jiahao/workspace/mvsplat/assets/evaluation_index_tartanair_nctx2.json'
        ),
        'tartianair4view' : (
            'tartianair',
            '/data0/jhshao/prometheus_benchmark/tartanair',
            '/home/jiahao/workspace/mvsplat/assets/evaluation_index_tartanair_nctx2.json'
        ),
        'tartianair5view' : (
            'tartianair', 
            '/data0/jhshao/prometheus_benchmark/tartanair',
            '/home/jiahao/workspace/mvsplat/assets/evaluation_index_tartanair_nctx6.json'
        ),
        'acid': (
            'acid',
            '/data1/yyb/datasets/acid',
            '/home/jiahao/workspace/mvsplat/assets/evaluation_index_acid.json'),
        're10k': (
            're10k',
            '/data0/datasets/re10k',
            '/home/jiahao/workspace/mvsplat/assets/evaluation_index_re10k.json'),

        'ablation_depth':(
            're10k',
            '/data0/datasets/re10k',
            '/home/jiahao/workspace/mvsplat/assets/evaluation_index_re10k.json')
    }
    
@dataclass
class Args:
    config: str = "configs_legacy/inference.yaml"
    ckpt: str = "pretrained/gsdecoder/epoch=383-step=383999.ckpt"
    task_name: str = 'tartianair2view'
    export_all: bool = True
    export_video: bool = False
    export_camera: bool = False
    export_ply: bool = False
    export_image: bool = True
    num_input_views: int = 4
    num_novel_views: int = 4
    render_size: int = 256
    out_dir: str = './workdir/paper/stage1'
    num_samples: int = 1
    gpu: int = 1
    seed: int = 42
    extra_filename: str = ''

def run_eval(args: Args):
    # print(args)
    lightning.seed_everything(args.seed)
    args.export_video = args.export_video or args.export_all
    args.export_camera = args.export_camera or args.export_all
    args.export_ply = args.export_ply or args.export_all
    args.export_image = args.export_image or args.export_all
    #extra_filename = args.extra_filename

    device = f'cuda:{args.gpu}'
    out_dir = Path(args.out_dir) / args.task_name
    os.makedirs(out_dir, exist_ok=True)


    render_size = args.render_size
    num_input_views, num_novel_views = args.num_input_views, args.num_novel_views
    extra_filename = args.extra_filename
    
    # load GS decoder
    params_decoder = torch.load(args.ckpt, weights_only=False, map_location=device)
    cfg_decoder = params_decoder['hyper_parameters']
    new_cfg = OmegaConf.load('configurations/global_env/local_workstation.yaml')
    cfg_decoder = merge_cfg(cfg_decoder, new_cfg)
    decoder_system = import_str(cfg_decoder['experiment']['training']['module'])(opt = cfg_decoder, mode = 'inference').eval().to(device).to(torch.bfloat16)
    decoder_system.load_state_dict(params_decoder['state_dict'], strict = False)

    dataset_name, root_dir, metadata_path = TASKS_[args.task_name]

    with open(metadata_path, 'r') as ff:
        eval_metadata = eval(ff.read())

    dataset = RealEstate10KDatasetEval(
          root_dir= os.path.join(root_dir, 'test'),
          dataset_name=dataset_name,
          img_size = render_size,
          )
    
    for scene_id, scene_meta in eval_metadata.items():
        target_ids, context_ids = scene_meta['contex'], scene_meta['target']
        batch = dataset.get_data(sceneid= scene_id, 
                        context_ids=context_ids, 
                        target_ids=target_ids)
        cameras, images = batch['cameras'], batch['images']
        with torch.amp.autocast("cuda", enabled=True):
            result = decoder_system.inference(
                cameras = cameras, 
                images = images,
                render_size = render_size
                # num_input_views = num_input_views,
                # render_size=render_size,
                # gs_decoder=decoder_system.model
                )
        
        save_dir =  out_dir / dataset_name / scene_id
        img_dir, depth_dir =  save_dir / 'image', save_dir / 'depth'
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        if args.export_image:
            # save raw rgb
            for i, fid in enumerate(target_ids):
                img_path = img_dir / f'{fid:06d}.png'
                depth_path = depth_dir / f'{fid:06d}.npy'
                # save depth
                image_save = postprocess_image(result['depths_gs_render'][i], -1, 1)
                imageio.save(img_path, image_save)
                depth_save = postprocess_image(result['images_gs_render'][i], 0, 1)
                np.save.save(depth_path, depths_save)

        if args.video:
            #TODO rendering video for teaser video
            pass 


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_eval(args)


