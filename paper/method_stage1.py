"""Infernece code of GS-Decoder system (scene-level GRM)"""
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
    config: str = "configs_legacy/inference.yaml",
    job_tag: str = 't383ckpt'
    ckpt: str = "pretrained/gsdecoder/epoch=383-step=383999.ckpt"
    export_all: bool = False
    export_video: bool = True
    export_camera: bool = True
    export_ply: bool = True
    export_image: bool = True
    use_ema: bool = True
    #num_refine_steps: int = 0
    out_dir: str = '/data1/yyb/PlatonicGen/workdir/paper/method/stage1'
    # num_samples: int = 1
    #use_3d_mode_every_m_steps: int = 10
    gpu: int = 0
    seed: int = 42
    extra_filename: str = ''

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

    # params = torch.load(args.ckpt, weights_only=True, map_location='cpu')

    if True:
        params = torch.load(args.ckpt, weights_only=False, map_location=device)
        # Build system based on cfg store in ckpt
        cfg = params['hyper_parameters']
        new_cfg = OmegaConf.load('configurations/global_env/local_workstation.yaml')
        cfg = merge_cfg(cfg, new_cfg)
        system = import_str(cfg['experiment']['training']['module'])(opt = cfg, mode = 'inference').eval().to(device)
        if args.use_ema:
            ema_state_dict = {}
            for k, v in params['state_dict'].items():
                if 'model_ema' in k:
                    ema_state_dict[k.replace('model_ema','model')] = v
                elif 'disp' in k:
                    ema_state_dict[k] = v
            system.load_state_dict(ema_state_dict, strict = False)
        else:
            system.load_state_dict(params['state_dict'])
    else:
        cfg = OmegaConf.load('configurations/global_env/local_workstation.yaml')
    render_size = 512
    num_input_views, num_novel_views = 4, 4


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
            result = system.inference(
                cameras = cameras, 
                images = images, 
                num_input_views = num_input_views,
                render_size=render_size,
                bg_color = (1,1,1)
                )
            
            gaussian = result['gaussians']

            save_dir = out_dir / filename
            img_dir = out_dir / 'image' / filename
            os.makedirs(img_dir, exist_ok = True)
            # img_dir.mkdir(exist_ok=True)
            if args.export_image:
                # save raw rgb
                img_path = img_dir / f'nv_pred.png'
                depths_nv_pred = postprocess_image(colorize_depth_maps(result['depths_nv_pred'], None, None,"Spectral"), 0, 1)
                images_nv_pred = postprocess_image(result['images_nv_pred'][0], -1, 1)
                images_iv_gt = postprocess_image(images[:num_input_views], -1, 1)
                raymaps_in = postprocess_image(result['rays_embeddings'][0,:,0:3], -1, 1)
                depth_iv_gt = postprocess_image(colorize_depth_maps(result['depths_in'][:,:,:1], None, None,"Spectral"), 0, 1)
                latents_in = postprocess_image(result['latents_in'][0,:,:3], -1, 1)
                latents_depth_in = postprocess_image(result['latents_in'][0,:,5:8], -1, 1)
                images_nv_gt = postprocess_image(images[num_input_views:], -1, 1)
                # depths_nv_pred_ = concatenate_images(depths_nv_pred)
                # images_nv_pred = concatenate_images(images_nv_pred)
                imageio.imsave(img_path,
                                np.concatenate((concatenate_images(depths_nv_pred),concatenate_images(images_nv_pred)), axis = 0))
                imageio.imsave(str(img_path).replace('nv_pred', 'nv_gt'),
                                concatenate_images(images_nv_gt))
                
                # for i in range(images_iv_gt.shape[0]):
                # imageio.imsave(str(img_path).replace('pred', 'indepth_gt.png'),concatenate_images(depth_iv_gt))
                # imageio.imsave(str(img_path).replace('pred', 'indepth_gt.png'),concatenate_images(depth_iv_gt))
                # imageio.imsave(str(img_path).replace('pred', 'z.png'),
                #                  concatenate_images(latents_in))
                # imageio.imsave(str(img_path).replace('pred', 'zd.png'),
                #                  concatenate_images(latents_depth_in))
                # save raw depth
                full_dir = img_dir / 'full'
                os.makedirs(full_dir, exist_ok=True)
                for i in range(4):
                    imageio.imsave(str(full_dir / f'in_rgb_gt_{i}.png'),images_iv_gt[i])
                    imageio.imsave(str(full_dir / f'in_depth_gt_{i}.png'),depth_iv_gt[i])
                    imageio.imsave(str(full_dir / f'in_raymap_{i}.png'),raymaps_in[i])            
                    imageio.imsave(str(full_dir / f'in_depth_latents_{i}.png'),latents_depth_in[i])
                    imageio.imsave(str(full_dir / f'in_rgb_latents_{i}.png'),latents_in[i])
                    imageio.imsave(str(full_dir / f'nv_rgb_pred_{i}.png'),images_nv_pred[i])
                    imageio.imsave(str(full_dir / f'nv_depth_pred_{i}.png'),depths_nv_pred[i])
                    imageio.imsave(str(full_dir / f'nv_rgb_gt_{i}.png'),images_nv_gt[i])

                    # imageio.imsave(str(full_dir / f'nv_depth_pred_{i}.png'),depthss_nv_pred)

            if args.export_ply:
                ply_dir = out_dir / 'ply' / filename
                os.makedirs(ply_dir, exist_ok=True)
                export_ply_for_gaussians(os.path.join(ply_dir, f'{filename}'), result['gaussians'])
            
            if args.export_video:
                video_dir = out_dir / 'video' / filename
                os.makedirs(video_dir, exist_ok=True)
                render_fn = lambda cameras, h=render_size, w=render_size: system.model.render(cameras[:num_input_views], result['gaussians'], h=h, w=w, bg_color=None)[:2]
                export_video(render_fn, video_dir , f'{filename}{extra_filename}', cameras[None,:num_input_views], device=device, fps=20, num_frames=80, render_size=render_size,
                             )


            # if args.export_ply:
            #     import plotly
            #     from pytorch3d.structures import Pointclouds
            #     from pytorch3d.vis.plotly_vis import plot_scene
            #     ply_dir = os.path.join(out_dir, 'ply')
            #     os.makedirs(ply_dir, exist_ok=True)
            #     pcd = export_ply_for_gaussians(
            #         os.path.join(ply_dir, f'{filename}{extra_filename}'), 
            #         result['gaussians'])
                
            #     pcd_mask = np.all(abs(pcd['pts']) < 10, axis = -1)
            #     pcd_pts, pcd_color = pcd['pts'][pcd_mask],  pcd['color'][pcd_mask]
            #     pcd = Pointclouds(
            #         points=torch.tensor(pcd_pts[None]), 
            #         features=torch.tensor(pcd_color[None])
            #     )
            #     fig = plotly_scene_visualization(camera_file=cameras, pcd=pcd)

            #     html_plot = plotly.io.to_html(fig, full_html=False, include_plotlyjs="cdn")
            #     HTML_TEMPLATE = HTML_TEMPLATE = """
            #     <html>
            #     <head>
            #         <meta charset="utf-8"/>
            #     </head>
            #     <body>
            #         <img src="data:image/png;charset=utf-8;base64,{input_view_encoded}" alt="Input View"/>
            #         <img src="data:image/png;charset=utf-8;base64,{novel_view_encoded}" alt="Novel View"/>
            #         {plotly_html}
            #     </body>
            #     </html>
            #     """
            #     images_nv_pred = postprocess_image(result['images_nv_pred'][0], -1, 1)
            #     images_iv_gt = postprocess_image(images[:num_input_views], -1, 1)
            #     # save input view gt
            #     s = io.BytesIO()
            #     fig = view_color_coded_images(images_iv_gt, c_range = [0,0.5])
            #     fig.suptitle('Input views', x=0.5, y=0.1)
            #     fig.savefig(s, format="png", bbox_inches="tight")
            #     input_image_encoded = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
            #     # save input view gt
            #     s = io.BytesIO()
            #     fig = view_color_coded_images(images_nv_pred, c_range = [0.5,1.0])
            #     fig.suptitle('Novel views', x=0.5, y=0.1)
            #     fig.savefig(s, format="png", bbox_inches="tight")
            #     novel_image_encoded = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
 
            #     html_path = args.out_dir/'html'/f'{filename}{extra_filename}.html'
            #     html_path.parent.mkdir(exist_ok=True)
            #     with open(html_path, "w") as f:
            #         s = HTML_TEMPLATE.format(
            #             # image_encoded=image_encoded,
            #             plotly_html=html_plot,
            #             input_view_encoded=input_image_encoded,
            #             novel_view_encoded=novel_image_encoded,
            #             # rendered_video_encoder=input_image_encoded
            #         )
            #         f.write(s)

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_inference(args)


