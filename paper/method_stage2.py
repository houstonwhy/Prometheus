"""Inference code of GS-Decoder system (scene-level GRM)"""
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
from prometheus.utils.io import export_ply_for_gaussians, import_str, export_video
from prometheus.utils.camera import sample_from_dense_cameras
from prometheus.utils.image_utils import colorize_depth_maps, postprocess_image
from prometheus.utils.visualization import plotly_scene_visualization
from einops import rearrange
import imageio 
# from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib.pyplot as plt
from prometheus.datasets import RealEstate10KDataset, UrbanGenDataset, DL3DV10KDataset, MVImgNetDataset

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
@dataclass
class Args:
    # Model and checkpoint settings
    config: str = "configs_legacy/inference.yaml"
    ckpt_path: str = "pretrained/gsdecoder/epoch=383-step=383999.ckpt"
    method_name: str = "legacy_full"
    use_ema: bool = True
    
    # Dataset settings
    dataset_name: str = "DL3DV10K"  # Options: DL3DV10K, MVImgNet, RealEstate10K, UrbanGen, ACID
    dataset_dir: str = "/path/to/dataset"
    img_size: int = 256
    num_input_views: int = 8
    num_novel_views: int = 8
    sample_rate: int = 8
    
    # Inference settings
    render_size: int = 512
    num_inference_steps: int = 100
    num_samples: int = 4
    cfg_type: str = "joint"  # Options: joint, text
    cfg_scale: float = 7.5
    cfg_rescale: float = 0.7
    
    # Export settings
    export_all: bool = False
    export_video: bool = True
    export_camera: bool = True
    export_ply: bool = True
    export_image: bool = True
    
    # Output settings
    out_dir: str = '/data1/yyb/PlatonicGen/workdir/paper/method/stage1'
    job_tag: str = 't383ckpt'
    extra_filename: str = ''
    
    # System settings
    gpu: int = 0
    seed: int = 42

def get_dataset(args):
    dataset_config = {
        "root_dir": args.dataset_dir,
        "img_size": args.img_size,
        "num_input_views": args.num_input_views,
        "num_novel_views": args.num_novel_views,
        "sample_rate": args.sample_rate,
        "debug": True
    }
    
    if args.dataset_name == "DL3DV10K":
        dataset_config["annotation_path"] = os.path.join(args.dataset_dir, "dl3dv_train.pkl")
        return DL3DV10KDataset(**dataset_config)
    elif args.dataset_name == "MVImgNet":
        dataset_config["annotation_path"] = os.path.join(args.dataset_dir, "mvimgnet_full_nocaption_new.pkl")
        dataset_config["normalized_cameras"] = True
        dataset_config["use_caption"] = False
        dataset_config["drop_text_p"] = 1.0
        return MVImgNetDataset(**dataset_config)
    elif args.dataset_name == "RealEstate10K":
        dataset_config["root_dir"] = os.path.join(args.dataset_dir, "test")
        dataset_config["normalized_cameras"] = True
        dataset_config["use_caption"] = False
        dataset_config["drop_text_p"] = 1.0
        return RealEstate10KDataset(**dataset_config)
    elif args.dataset_name == "UrbanGen":
        dataset_config["annotation_path"] = os.path.join(args.dataset_dir, "urbangen_full_nocaption.pkl")
        dataset_config["use_caption"] = False
        dataset_config["drop_text_p"] = 1.0
        dataset_config["scene_scale_threshold"] = 20
        return UrbanGenDataset(**dataset_config)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

def run_inference(args: Args):
    print(args)
    args.export_video = args.export_video or args.export_all
    args.export_camera = args.export_camera or args.export_all
    args.export_ply = args.export_ply or args.export_all
    args.export_image = args.export_image or args.export_all
    #extra_filename = args.extra_filename
    # args.out_dir = Path(args.out_dir)
    # args.out_dir.mkdir(exist_ok=True)
    torch.backends.cudnn.benchmark = True

    device = f'cuda:{args.gpu}'
    extra_filename = args.extra_filename
    lightning.seed_everything(args.seed)


    method_name = 'legacy_full'
    ckpt_path = 'outputs/ckpts/prometheus_k8s_mvldm_viewcond_exp_mvldm_dataset_mvldm_viewcond_mvldm_viewcond_exp_exp11a_mvldm_highnoise/epoch=12-step=13000.ckpt'
    params = torch.load(ckpt_path, weights_only=False, map_location=device)
    cfg = params['hyper_parameters']
    
    use_ema = args.use_ema

    new_cfg = OmegaConf.load('configurations/global_env/local_workstation.yaml')
    cfg = merge_cfg(cfg, new_cfg)
    from prometheus.utils.io import import_str
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
        system.load_state_dict(params['state_dict'], strict = False)
    gsdecoder_ext = None
    # if gsdecoder_ext is not None:
    #     system.gsdecoder = gsdecoder_ext
    # Re config schdeuler for legacy ckpts -> set p_data=0.5
    if method_name in ['no_sv', 'low_noise','legacy_full_nohigh']:
        system.high_noise_level = False
        system.configure_noise_scheduler(0.5)
    elif method_name in ['legacy_full', 'rescale0']:
        print('use legacy_full')
        system.high_noise_level = True
        system.configure_noise_scheduler(0.5)
        cfg_type, cfg_scale = 'joint', 7.5
        cfg_rescale = 0.0 if method_name == 'rescale0' else 0.7
    else:
        system.high_noise_level = True
        system.configure_noise_scheduler(1.)
    
    if method_name == 'text_cfg':
        cfg_type, cfg_scale = 'text', 7.5
    elif method_name == 'nodepth':
        cfg_type, cfg_scale = 'joint', 3.5
    elif method_name == 'legacy_full_nohigh':
        cfg_type, cfg_scale = 'joint', 7.5
        cfg_rescale = 0.0
    # elif method_name == 'rescale0':
    #     # cfg_type, cfg_scale = 'joint', 7.5
    #     cfg_rescale = 0.0


    render_size = 512
    num_input_views, num_novel_views = 8, 8

    
    # dataset = import_str(cfg['experiment']['training']['module'])(opt = cfg, mode = 'inference').eval().to(device).to(torch.bfloat16)
    dataset = DL3DV10KDataset(
                root_dir=cfg.DL3DV_PATH,
                annotation_path=os.path.join(cfg.DL3DV_MATADATA_PATH,'dl3dv_train.pkl'),
                #annotation_meta: metadata/dl3dv_train.pkl
                img_size=256,
                num_input_views=4,
                num_novel_views=4,
                sample_rate=8,
                debug = True
            )
    # dataset = MVImgNetDataset(
    #             root_dir=cfg.MVImgNet_PATH,
    #             annotation_path=os.path.join(cfg.MVImgNet_MATADATA_PATH,'mvimgnet_full_nocaption_new.pkl'),
    #             img_size = 256,
    #             normalized_cameras=True,
    #             use_caption = False,
    #             drop_text_p = 1.,
    #             num_input_views = num_input_views,
    #             num_novel_views = num_novel_views,
    #             sample_rate = 10000,
    #             debug = True
    #     )
    # dataset = RealEstate10KDataset(
    #       root_dir= os.path.join(cfg.RealEstate10K_PATH, 'test'),
    #       #root_dir= '/input/datasets/re10k/train',
    #       img_size = 256,
    #       normalized_cameras = True,
    #       use_caption = False,
    #       drop_text_p = 1.,
    #       num_input_views = num_input_views,
    #       num_novel_views = num_novel_views,
    #       sample_rate = 10000,
    #       debug = False
    #       )
    # dataset = UrbanGenDataset(
    #         root_dir = cfg.UrbanGen_PATH,
    #         annotation_path = os.path.join(cfg.UrbanGen_MATADATA_PATH,'urbangen_full_nocaption.pkl'),
    #         img_size = 256,
    #         use_caption = False,
    #         drop_text_p=1.,
    #         num_input_views = num_input_views,
    #         num_novel_views = num_novel_views,
    #         sample_rate= 8,
    #         scene_scale_threshold=20,
    #         debug = False
    #         )
    # dataset = RealEstate10KDataset(
    #       root_dir= os.path.join(cfg.ACID_PATH, 'train'),
    #       #root_dir= '/input/datasets/re10k/train',
    #       dataset_name='ACID',
    #       img_size = 256,
    #       drop_text_p = 1.,
    #       num_input_views = num_input_views,
    #       num_novel_views = num_novel_views,
    #       sample_rate = 100,
    #       scene_scale_threshold=1.,
    #       debug = False)
    job_tag = args.job_tag
    if args.use_ema:
        job_tag += '_ema'
    out_dir = Path(args.out_dir) / job_tag
    num_samples = 4
    
    with torch.no_grad():
        if True:
            # re10k
            #
            if dataset.dataset_name == 'MVImgNet':
                target_scenes = [i for i, ff in enumerate(dataset.items) if '_'.join(dataset.items[0].split('/')[-2:]) == '99_1001290e']
            elif dataset.dataset_name == 'RealEstate10K':
                target_scenes = [i for i, ff in enumerate(dataset.items) if ff == '6558c5f10d45a929']
            elif dataset.dataset_name == 'DL3DV10K':
                target_scenes = [i for i, ff in enumerate(dataset.items) if ff =='3e6a66ea9536530d688aaaf63fda9229d488a446ba256d113d03f29b0b2e7647_1']
            # acid
            elif dataset.dataset_name == 'ACID':
                target_scenes = [i for i, ff in enumerate(dataset.items) if ff == 'bef19afae1aabdbd']
            else:
                target_scenes = range(len(dataset))
        else:
            target_scenes = range(len(dataset))
        for idx in target_scenes:
        # for idx in range(len(dataset)):
            data = dataset[idx]
            cameras, images = data['cameras_mv'], data['images_mv']
            filename = '_'.join([data['dataset_name_mv'], data['scene_name_mv'].replace('/', '_')])
            if True:
            # for _, scene_path in enumerate(scene_list):
                # if str(out_dir / filename / f"{i}" / filename)

                # for idx in range(len(dataset)):
                # with open(scene_path, 'rb') as fp:
                #     data = pkl.load(fp)
                cameras, text = torch.tensor(data['cameras_mv']).to(device), data['text_mv']
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
                            num_inference_steps=100, 
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
                            # if args.num_refine_steps > 0:
                            #     os.makedirs(os.path.join(args.out_dir, f'ply_sds{args.num_refine_steps}'), exist_ok=True)
                            #     export_ply_for_gaussians(os.path.join(args.out_dir, f'ply_sds{args.num_refine_steps}', f'{filename}{extra_filename}'), result['refined_gaussians'])
                    if args.export_video:
                        if hasattr(system, 'gsdecoder'):
                            gsdecoder = system.gsdecoder
                        else:
                            gsdecoder = gsdecoder_ext
                        video_frame_dir = str(out_dir / filename / f"{i}" / filename)
                        os.makedirs(video_frame_dir,exist_ok=True)
                        render_fn = lambda cameras, h=render_size, w=render_size: gsdecoder.render(cameras, result['gaussians'], h=h, w=w, bg_color=(1,1,1))[:2]
                        full_images, full_depths = export_video(render_fn, save_dir , filename, cameras, device=device, fps=30, num_frames=120, render_size=render_size)
                        for j, pp in enumerate(full_images):
                                imageio.imsave(os.path.join(video_frame_dir, str(j).zfill(6) + ".png"), pp)
                



if __name__ == "__main__":
    args = tyro.cli(Args)
    run_inference(args)


