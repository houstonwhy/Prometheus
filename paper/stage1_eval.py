"""Infernece code Stage1 evalaution Sce4.1.1"""
# pylint: disable=import-error
import os
import io
import base64
import json
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
        
        'tartianair4view_hard' : (
            'tartianair',
            '/data0/jhshao/prometheus_benchmark/tartanair',
            '/home/jiahao/workspace/mvsplat/assets/evaluation_index_tartanair_nctx4_20_35_0.3.json'
        ),

        'tartianair4view_nightmare' : (
            'tartianair',
            '/data0/jhshao/prometheus_benchmark/tartanair',
            '/home/jiahao/workspace/mvsplat/assets/evaluation_index_tartanair_nctx4_30_45_0.3.json'
        ),

        'tartianair4view_legend' : (
            'tartianair',
            '/data0/jhshao/prometheus_benchmark/tartanair',
            '/home/jiahao/workspace/mvsplat/assets/evaluation_index_tartanair_nctx4_40_50_0.3.json'
        ),
        # ----------
        'tartianair4view_easyvideo' : (
            'tartianair',
            '/data0/jhshao/prometheus_benchmark/tartanair',
            '/home/jiahao/workspace/mvsplat/assets/evaluation_index_tartanair_easy_video.json'
        ),

        'tartianair4view_hardvideo' : (
            'tartianair',
            '/data0/jhshao/prometheus_benchmark/tartanair',
            '/home/jiahao/workspace/mvsplat/assets/evaluation_index_tartanair_hard_video.json'
        ),

        'tartianair4view_mediumvideo' : (
            'tartianair',
            '/data0/jhshao/prometheus_benchmark/tartanair',
            '/home/jiahao/workspace/mvsplat/assets/evaluation_index_tartanair_medium_video.json'
        ),

        # aa
        'tartianair6view' : (
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
        # #TODO Ablation depth and multiple dataset
        # 'ablation_depth':(
        #     're10k',
        #     '/data0/datasets/re10k',
        #     '/home/jiahao/workspace/mvsplat/assets/evaluation_index_re10k.json')
    }

METHODS_ = {
    'full':'pretrained/gsdecoder/epoch=383-step=383999.ckpt',
    'full_randombg':'outputs/ckpts/prometheus_k8s_gsdecoder_exp_gsdecoder_dataset_full_gsdecoder_dit_gsdecoder_exp_exp06j_mask/epoch=482-step=482009.ckpt',
    'nodsv':'outputs/ckpts/prometheus_k8s_gsdecoder_exp_gsdecoder_dataset_full_gsdecoder_dit_gsdecoder_exp_exp06j_mask/epoch=482-step=482009.ckpt',
    'nodepth':'outputs/ckpts/prometheus_k8s_gsdecoder_exp_gsdecoder_dataset_full_gsdecoder_dit_gsdecoder_exp_exp09b_gsdecoder_ablationdepth/epoch=86-step=87000.ckpt',
    're10konly':'outputs/ckpts/prometheus_k8s_gsdecoder_exp_gsdecoder_dataset_re10k_gsdecoder_dit_gsdecoder_exp_exp06k_re10k/epoch=471-step=471999.ckpt'
}
@dataclass
class Args:
    config: str = "configs_legacy/inference.yaml"

    # out_dir: str = './workdir/paper/stage1_black_06h'
    # out_dir: str = './workdir/paper/stage1_legacy_06h'
    # ckpt: str = "outputs/ckpts/prometheus_k8s_gsdecoder_exp_gsdecoder_dataset_full_gsdecoder_dit_gsdecoder_exp_exp06h_objaverse/epoch=215-step=215999.ckpt"
    # out_dir: str = './workdir/paper/stage1_raw_06h'
    # ckpt: str = "pretrained/gsdecoder/epoch=383-step=383999.ckpt"
    out_dir: str = './workdir/paper/stage1/'
    method_name: str = 'nodepth'
    # ckpt: str = "outputs/ckpts
    tasks: str = 'tartianair4view_hard,tartianair4view'
    # tasks: str = 'tartianair4view,re10k,acid,tartianair2view,tartianair6view,'
    export_all: bool = True
    export_video: bool = False
    export_camera: bool = False
    export_ply: bool = False
    export_image: bool = True
    num_input_views: int = 4
    num_novel_views: int = 4
    render_size: int = 256
    
    num_samples: int = 1
    gpu: int = 0
    seed: int = 42
    extra_filename: str = ''
    bg_color: tuple = (1,1,1)

def run_eval(args: Args):
    # print(args)
    lightning.seed_everything(args.seed)
    args.export_video = args.export_video or args.export_all
    args.export_camera = args.export_camera or args.export_all
    args.export_ply = args.export_ply or args.export_all
    args.export_image = args.export_image or args.export_all

    device = f'cuda:{args.gpu}'

    render_size = args.render_size
    bg_color = args.bg_color

    method_name = args.method_name
    ckpt_path = METHODS_[method_name]
    # tartianair2view/prometheus/frames/abandonedfactory_night_P000/color
    # load GS decoder
    if False:
        with open(ckpt_path, 'rb') as f:
                os.fsync(f.fileno())
                buffer = io.BytesIO(f.read())
        params_decoder = torch.load(buffer, weights_only=False, map_location=device)
    else:
        params_decoder = torch.load(ckpt_path, weights_only=False, map_location=device)
    # params_decoder = torch.load(args.ckpt, weights_only=False, map_location=device)
    cfg_decoder = params_decoder['hyper_parameters']
    new_cfg = OmegaConf.load('configurations/global_env/local_workstation.yaml')
    cfg_decoder = merge_cfg(cfg_decoder, new_cfg)
    decoder_system = import_str(cfg_decoder['experiment']['training']['module'])(opt = cfg_decoder, mode = 'inference').eval().to(device).to(torch.bfloat16)
    decoder_system.load_state_dict(params_decoder['state_dict'], strict = True)

    task_list = args.tasks.split(',')
    for task in task_list:
        out_dir = Path(args.out_dir) / method_name / task
        os.makedirs(out_dir, exist_ok=True)
        dataset_name, root_dir, metadata_path = TASKS_[task]

        with open(metadata_path, 'r') as ff:
            eval_metadata = json.load(ff)

        print(f"Start processing {task}, contains {len(eval_metadata)} seqs in total")

        dataset = RealEstate10KDatasetEval(
            root_dir= os.path.join(root_dir, 'test'),
            dataset_name=dataset_name,
            img_size = render_size,
            )
        for scene_id, scene_meta in tqdm.tqdm(eval_metadata.items()):
            if scene_meta is None:
                continue
            context_ids, target_ids = scene_meta['context'], scene_meta['target']
            batch = dataset.get_data(sceneid= scene_id, 
                            context_ids=context_ids, 
                            target_ids=target_ids)
            # context + target
            cameras, images = batch['cameras'], batch['images']
            with torch.amp.autocast("cuda", enabled=True):
                result = decoder_system.inference(
                    cameras = cameras, 
                    images = images,
                    num_input_views = len(context_ids),
                    render_size = render_size,
                    bg_color = bg_color,
                    )
            
            save_dir =  out_dir / 'prometheus' / 'frames' / scene_id
            img_dir, depth_dir =  save_dir / 'color', save_dir / 'depth'
            imggt_dir = save_dir / 'color_gt'
            error_color_dir = save_dir / 'color_error'
            depth_color_dir =  save_dir / 'depth_color'
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(imggt_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(depth_color_dir, exist_ok=True)
            os.makedirs(error_color_dir, exist_ok=True)

            if args.export_image:
                # save raw rgb
                images_to_save = postprocess_image(result['images_nv_pred'][0], -1, 1,  return_PIL = True)
                imagesgt_to_save = postprocess_image(images[len(context_ids):], -1, 1,  return_PIL = True)
                error_to_save = postprocess_image(abs(images[len(context_ids):] - result['images_nv_pred'][0].cpu()), 0, 2,  return_PIL = True)
                depths_to_save = result['depths_nv_pred'][0][:,0].cpu().numpy()
                depthcolor_to_save = postprocess_image(colorize_depth_maps(result['depths_nv_pred'][0]), 0, 1)
                for i, fid in enumerate(target_ids):
                    img_path = img_dir / f'{fid:06d}.png'
                    imggt_path = imggt_dir / f'{fid:06d}.png'
                    error_color_path = error_color_dir / f'{fid:06d}.png'
                    depth_path = depth_dir / f'{fid:06d}.npy'
                    depth_color_path = depth_color_dir/ f'{fid:06d}.png'
                    # save depth
                    imageio.imsave(str(imggt_path), imagesgt_to_save[i])
                    imageio.imsave(str(img_path), images_to_save[i])
                    imageio.imsave(str(depth_color_path), depthcolor_to_save[i])
                    
                    imageio.imsave(str(error_color_path), error_to_save[i])
                    with open(depth_path, 'wb') as fp:
                        np.save(fp, depths_to_save[i])
            # if args.video:
            #     #TODO rendering video for teaser video
            #     pass 

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_eval(args)


