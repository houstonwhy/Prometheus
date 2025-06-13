from typing import Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler
import numpy as np
import os
import einops
import copy
from torchvision.utils import save_image
import random
from diffusers import DDIMScheduler
from dual3dgs import Dual3DGSModel

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning
import warnings
warnings.filterwarnings("ignore")
import tqdm

# from datasets.mixing import MixingDataset

from transformers import pipeline
from torchmetrics import PearsonCorrCoef

from utils import ssim, depth_loss, get_random_cameras, inverse_sigmoid, import_str

import lpips

from utils import matrix_to_square

def update_average(model_tgt, model_src, beta=0.995):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert (p_src is not p_tgt)
            p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

        buffer_dict_src = dict(model_src.named_buffers())
        for p_name, p_tgt in model_tgt.named_buffers():
            p_src = buffer_dict_src[p_name]
            assert (p_src is not p_tgt)
            p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

class EMANorm(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.register_buffer('magnitude_ema', torch.ones([]))
        self.beta = beta

    def forward(self, x):
        if self.training:
            magnitude_cur = x.detach().to(torch.float32).square().mean()
            self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.beta))
        input_gain = self.magnitude_ema.rsqrt()
        x = x.mul(input_gain)
        return x

class Dual3DSystem(LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters(opt)
        self.opt = opt
        
        self.image_size = self.opt.network.image_size
        self.latent_size = self.opt.network.latent_size
        self.latent_channel = self.opt.network.latent_channel
        
        self.model = Dual3DGSModel(opt)

        # pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

        # self.disp_fn = pipe.model.eval().requires_grad_(False)

        # self.register_buffer("disp_image_mean", torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), persistent=False)
        # self.register_buffer("disp_image_std", torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), persistent=False)

        self.lpips_fn = lpips.LPIPS(net='vgg').eval().requires_grad_(False)

        # self.scheduler = DDIMScheduler.from_pretrained(
        #     self.opt.network.sd_model_key, subfolder="scheduler"
        # )
        # self.scheduler.config.prediction_type = 'sample'
        # self.scheduler = DDIMScheduler(beta_schedule='squaredcos_cap_v2', prediction_type="sample", clip_sample=False, steps_offset=9)

        self.scheduler = DDIMScheduler(beta_schedule='scaled_linear', beta_start=0.00085, beta_end=0.012, prediction_type="sample", clip_sample=False, steps_offset=9, rescale_betas_zero_snr=True, set_alpha_to_one=True)

        self.register_buffer("alphas_cumprod", self.scheduler.alphas_cumprod, persistent=False)
     
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = 0
        self.max_step = int(self.num_train_timesteps)
        
        self.num_input_views = self.opt.network.num_input_views
        self.num_novel_views = self.opt.network.num_novel_views

        self.model_ema = copy.deepcopy(self.model).requires_grad_(False)

        self.refiner = None
        
    def configure_optimizers(self):
        params = []
        for p in self.model.parameters():
            if p.requires_grad: params.append(p)
        optimizer = torch.optim.AdamW(params, lr=self.opt.training.learning_rate / self.opt.training.accumulate_grad_batches, weight_decay=self.opt.training.weight_decay, betas=self.opt.training.betas)
        return optimizer
    
    @torch.amp.autocast("cuda", enabled=False)
    def prepare_data_for_different_task(self, latents_noisy, texts, cameras, t, input_latents, task='text_to_3d'):
        B, N = t.shape    

        texts = list(texts)
        if task == 'text_to_3d':
            masks = torch.zeros(B, device=self.device)
        elif task == 'image_to_3d':
            masks = torch.ones(B, device=self.device)
        else:
            masks = torch.randint(0, 2, (B,), device=self.device)

        for i in range(B):
            if masks[i].item() == 0:
                if random.random() < self.opt.training.text_to_3d_drop_text_p:
                    texts[i] = ''
            else:
                latents_noisy[i, :1] = input_latents[i, :1]
                t[i, :1] = -1

                # c2ws = cameras[i, :, :12].reshape(-1, 3, 4)
                # ref_w2c = torch.inverse(matrix_to_square(c2ws[:1]))
                # c2ws = (ref_w2c.repeat(c2ws.shape[0], 1, 1) @ matrix_to_square(c2ws))[:,:3,:]

                # cameras[i, :, :12] = c2ws.reshape(-1, 12)

                if random.random() < self.opt.training.image_to_3d_drop_text_p:
                    texts[i] = ''
                if random.random() < self.opt.training.image_to_3d_drop_image_p:
                    latents_noisy[i, :1] = torch.zeros_like(latents_noisy[i, :1])

        return latents_noisy, texts, cameras, t

    # @torch.no_grad()
    # def get_depth_gt(self, x):
    #     B, N, C, H, W = x.shape
    #     x = x.flatten(0, 1)
    #     x = F.interpolate(x, size=(518, 518), align_corners=False, mode='bicubic')
    #     disps = self.disp_fn(((x + 1)/2 - self.disp_image_mean) / self.disp_image_std).predicted_depth.unsqueeze(1)
    #     disps = F.interpolate(disps, size=(H, W), align_corners=False, mode='bilinear')

    #     disps_flatten = disps.flatten(1, -1)
    #     min_disps_flatten = disps_flatten.min(dim=1)[0].reshape(B, 1, 1, 1)
    #     max_disps_flatten = disps_flatten.max(dim=1)[0].reshape(B, 1, 1, 1)
    #     disps = (disps - min_disps_flatten) / (max_disps_flatten - min_disps_flatten + 1e-5)

    #     disps = disps.unflatten(0, (B, N))

    #     return 1 / (disps + 1e-2)
    
    def add_noise(self, x, noise, t):
        x_noisy = self.scheduler.add_noise(x, noise, t)
        return x_noisy
        
    def training_step(self, batch, _):
        self.lpips_fn.eval().requires_grad_(False)
        # self.disp_fn.eval().requires_grad_(False)
        self.model.train()
        update_average(self.model_ema, self.model)
        
        images_original, cameras, texts, images2d, _texts2d  = batch

        loss_total = 0
        # 2d part:
        # images2d = images2d.flatten(0, 1).unsqueeze(1)
        # texts2d = []
        # for text2d in _texts2d:
        #     texts2d += text2d
        # B, _, C, H, W = images2d.shape    

        # with torch.no_grad():
        #     input_latents2d = self.model.encode_image(images2d)
        #     # depths2d = self.get_depth_gt(images2d)

        # t = torch.randint(0, self.num_train_timesteps, (B,), dtype=torch.long, device=self.device)
        # latents_noisy = self.add_noise(input_latents2d, torch.randn_like(input_latents2d), t)
        # t = t.unsqueeze(1)
        # text_embeddings = self.model.encode_text(texts2d)
        # random_cameras = cameras[:, :1].repeat(B//cameras.shape[0], 1, 1)
        # latents_pred, gaussians2d = self.model.denoise(latents_noisy, text_embeddings, t, random_cameras, return_3d=True)
        
        # loss_sv_latent_mse = F.mse_loss(input_latents2d, latents_pred)

        # loss_total += loss_sv_latent_mse * self.opt.losses.lambda_sv_latent_mse

        # with torch.amp.autocast("cuda", enabled=False):    
        #     images2d_pred, depths2d_pred, _, reg_losses2d, _ = self.model.render(random_cameras, gaussians2d, h=self.image_size, w=self.image_size)

        #     images2d = images2d.reshape(B, -1, self.image_size, self.image_size)
        #     images2d_pred = images2d_pred.reshape(B, -1, self.image_size, self.image_size)

        #     loss_sv_image_mse = F.mse_loss(images2d, images2d_pred)
        #     loss_sv_image_lpips = self.lpips_fn(images2d, images2d_pred).mean()

        #     loss_total += loss_sv_image_mse * self.opt.losses.lambda_sv_image_mse
        #     loss_total += loss_sv_image_lpips * self.opt.losses.lambda_sv_image_lpips

        #     # depths2d = depths2d.reshape(B, self.image_size * self.image_size)
        #     # depths2d_pred = depths2d_pred.reshape(B, self.image_size * self.image_size)

        #     # loss_sv_depth = depth_loss(depths2d / 100, depths2d_pred / 100)

        #     # loss_total += loss_sv_depth * self.opt.losses.lambda_sv_depth

        #     self.log('losses/single_view/loss_sv_latent_mse', loss_sv_latent_mse, sync_dist=True)
        #     self.log('losses/single_view/loss_sv_image_mse', loss_sv_image_mse, sync_dist=True)
        #     self.log('losses/single_view/loss_sv_image_lpips', loss_sv_image_lpips, sync_dist=True)

        #     # self.log('losses/single_view/loss_sv_depth', loss_sv_depth, sync_dist=True)

        # 3d part:
        B, N, C, H, W = images_original.shape    
        images_original = F.interpolate(images_original.flatten(0, 1), self.image_size, mode='bilinear', align_corners=False).unflatten(0, (B, N))
        input_views = images_original[:, :self.num_input_views]

        with torch.no_grad():
            input_latents = self.model.encode_image(input_views)
        t = torch.randint(0, self.num_train_timesteps, (B,), dtype=torch.long, device=self.device)
        latents_noisy = self.add_noise(input_latents, torch.randn_like(input_latents), t)
        t = t.unsqueeze(1).repeat(1, self.num_input_views)

        latents_noisy, texts, cameras, t = self.prepare_data_for_different_task(latents_noisy, texts, cameras, t, input_latents)

        text_embeddings = self.model.encode_text(texts)
        latents_pred, gaussians = self.model.denoise(latents_noisy, text_embeddings, t, cameras=cameras[:, :self.num_input_views])
        
        loss_mv_latent_mse = F.mse_loss(input_latents, latents_pred)
        # novel view synthesis
        with torch.amp.autocast("cuda", enabled=False):    
            nv_images = images_original[:, self.num_input_views:]
            nv_images_pred, _, _, reg_losses, states = self.model.render(cameras[:, self.num_input_views:], gaussians, h=self.image_size, w=self.image_size)

        nv_images = nv_images.reshape(B * self.num_novel_views, -1, self.image_size, self.image_size)
        nv_images_pred = nv_images_pred.reshape(B * self.num_novel_views, -1, self.image_size, self.image_size)

        loss_mv_image_mse = F.mse_loss(nv_images_pred, nv_images)
        loss_mv_image_lpips = self.lpips_fn(nv_images_pred, nv_images).mean()

        loss_total += loss_mv_latent_mse * self.opt.losses.lambda_mv_latent_mse
        loss_total += loss_mv_image_mse * self.opt.losses.lambda_mv_image_mse
        loss_total += loss_mv_image_lpips * self.opt.losses.lambda_mv_image_lpips

        loss_total += reg_losses['loss_reg_opacity'] * self.opt.losses.lambda_reg_opacity
        loss_total += reg_losses['loss_reg_scales'] * self.opt.losses.lambda_reg_scales

        self.log('losses/3d/loss_mv_latent_mse', loss_mv_latent_mse, sync_dist=True)
        self.log('losses/3d/loss_mv_image_mse', loss_mv_image_mse, sync_dist=True)
        self.log('losses/3d/loss_mv_image_lpips', loss_mv_image_lpips, sync_dist=True)

        for key, value in reg_losses.items():                 
            self.log(f'losses/reg/{key}', value, sync_dist=True)
            
        for key, value in states.items():
            self.log(f'states/{key}', value, sync_dist=True)
            
        return loss_total
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        images_original, cameras, texts, images2d, _texts2d  = batch

        # # 2d part:
        images2d = images2d.flatten(0, 1).unsqueeze(1)
        texts2d = []
        for text2d in _texts2d:
            texts2d += text2d
        B, _, C, H, W = images2d.shape    

        
        input_latents2d = self.model.encode_image(images2d)
        t = torch.randint(0, self.num_train_timesteps, (B,), dtype=torch.long, device=self.device)
        latents_noisy = self.add_noise(input_latents2d, torch.randn_like(input_latents2d), t)
        t = t.unsqueeze(1)
        text_embeddings = self.model.encode_text(texts2d)
        random_cameras = cameras[:, :1].repeat(B//cameras.shape[0], 1, 1)
        latents_pred, gaussians2d = self.model.denoise(latents_noisy, text_embeddings, t, random_cameras, return_3d=True)

        # depths2d = self.get_depth_gt(images2d)

        with torch.amp.autocast("cuda", enabled=False):    
            images2d_pred, depths2d_pred, _, _, _ = self.model.render(random_cameras, gaussians2d, h=self.image_size, w=self.image_size)

        self.logger.experiment.add_image(f"2d_gt/{int(batch_idx)}", (images2d.flatten(0, 1) + 1) / 2, self.global_step, dataformats='NCHW')
        self.logger.experiment.add_image(f"2d/{int(batch_idx)}", (images2d_pred.flatten(0, 1) + 1) / 2, self.global_step, dataformats='NCHW')

        # depths2d = depths2d.reshape(B, H * W)
        # depths2d_pred = depths2d_pred.reshape(B, H * W)
        # self.logger.experiment.add_image(f"2d_depth_gt/{int(batch_idx)}", 
        #     (1 / depths2d.reshape(B, 1, H, W)).clamp(0, 1), self.global_step, dataformats='NCHW')

        # 3d part
        B, N, C, H, W = images_original.shape    
        images_original = F.interpolate(images_original.flatten(0, 1), self.image_size, mode='bilinear', align_corners=False).unflatten(0, (B, self.num_input_views + self.num_novel_views))
        input_views = images_original[:, :self.num_input_views]
        input_latents = self.model.encode_image(input_views)

        t = torch.randint(0, self.num_train_timesteps, (B,), dtype=torch.long, device=self.device)
        latents_noisy = self.add_noise(input_latents, torch.randn_like(input_latents), t)
        t = t.unsqueeze(1).repeat(1, self.num_input_views)

        latents_noisy, texts, cameras, t = self.prepare_data_for_different_task(latents_noisy, texts, cameras, t, input_latents)

        text_embeddings = self.model.encode_text(texts)
        latents_pred, gaussians = self.model.denoise(latents_noisy, text_embeddings, t, cameras[:, :self.num_input_views])
        
        # novel view synthesis
        with torch.amp.autocast("cuda", enabled=False):    
            nv_images = images_original[:, self.num_input_views:]
            nv_images_pred, _, _, _, _ = self.model.render(cameras[:, self.num_input_views:], gaussians, h=self.image_size, w=self.image_size)
        
        self.logger.experiment.add_image(f"input_views/{int(batch_idx)}", (input_views.flatten(0, 1) + 1) / 2, self.global_step, dataformats='NCHW')
        
        self.logger.experiment.add_image(f"novel_view_gt/{int(batch_idx)}", (nv_images.flatten(0, 1) + 1) / 2, self.global_step, dataformats='NCHW')
        
        self.logger.experiment.add_image(f"novel_view/{int(batch_idx)}", (nv_images_pred.flatten(0, 1) + 1) / 2, self.global_step, dataformats='NCHW')
        return 

    def inference_one_step(self, cameras, latents_noisy, text_embeddings, uncond_text_embeddings, t, guidance_scale=10, use_3d_mode=True):
        _latents_noisy = latents_noisy.clone()
        B, N, _, _ ,_ = latents_noisy.shape

        _t = t[..., None].repeat(1, N)

        uncond_latents_noisy = latents_noisy.clone()

        uncond_t = _t.clone()

        if use_3d_mode:

            latents_noisy = latents_noisy
            cameras = cameras
            text_embeddings = text_embeddings
            tt = _t

            # latents_noisy = torch.cat([latents_noisy, uncond_latents_noisy], 0)
            # cameras = torch.cat([cameras, cameras], 0)
            # text_embeddings = torch.cat([text_embeddings, uncond_text_embeddings], 0)
            # tt = torch.cat([_t, uncond_t], 0)

            B, N = latents_noisy.shape[:2]
            _, gaussians = self.model.denoise(latents_noisy, text_embeddings, tt, cameras)

            images_pred, _, _, _, _ = self.model.render(cameras, gaussians, h=self.image_size, w=self.image_size)

            _latents_pred = self.model.encode_image(images_pred)

            # latents_pred, uncond_latents_pred = _latents_pred.chunk(2, dim=0)
            # _latents_pred = (latents_pred - uncond_latents_pred) * guidance_scale + uncond_latents_pred

            # if task == 'image_to_3d':
            #     _latents_pred[:, :1] = latents[:, :1]

            latents_less_noisy = self.scheduler.step(_latents_pred.cpu(), t.cpu(), _latents_noisy.cpu(), eta=1).prev_sample.to(self.device)
            
        else:  
            num_views = None
            cameras = torch.cat([cameras, cameras], 0)

            latents_noisy = torch.cat([latents_noisy, uncond_latents_noisy], 0)
            text_embeddings = torch.cat([text_embeddings, uncond_text_embeddings], 0)
            tt = torch.cat([_t, uncond_t], 0)

            latents2d_pred = self.model.denoise(latents_noisy, text_embeddings, tt, cameras, return_3d=False, num_views=num_views)
            
            latents_pred, uncond_latents_pred = latents2d_pred.chunk(2, dim=0)
            _latents_pred = (latents_pred - uncond_latents_pred) * guidance_scale + uncond_latents_pred

            # if task == 'image_to_3d':
            #     _latents_pred[:, :1] = latents[:, :1]
        
            latents_less_noisy = self.scheduler.step(_latents_pred.cpu(), t.cpu(), _latents_noisy.cpu(), eta=0).prev_sample.to(self.device)

        if use_3d_mode:
            return latents_less_noisy, {"gaussians": gaussians, "images_pred": images_pred}
        else:
            return latents_less_noisy, {'latents_pred': _latents_pred}
    
    @torch.no_grad()
    def inference(self, cameras, text, dense_cameras=None, num_inference_steps=100, num_refine_steps=0, guidance_scale=7.5, use_3d_mode_every_m_steps=10, negative_text=""):
        B, N = cameras.shape[:2]
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = self.scheduler.timesteps

        latents_noisy = torch.randn(B, N, self.latent_channel, self.latent_size, self.latent_size, device=self.device)

        # text_embeddings = self.model.encode_text([(text + ' 3D scene.') if is_scene else text])
        text_embeddings = self.model.encode_text([text])
        uncond_text_embeddings =  self.model.encode_text([negative_text]).repeat(B, 1, 1)

        assert use_3d_mode_every_m_steps != 1, "use_3d_mode_every_m_steps can not be 1"

        if use_3d_mode_every_m_steps == -1:
            guidance_scale = guidance_scale
        else:
            guidance_scale = guidance_scale * use_3d_mode_every_m_steps / (use_3d_mode_every_m_steps - 1)

        results = []
        for i, t in tqdm.tqdm(enumerate(timesteps), total=len(timesteps)):
            if use_3d_mode_every_m_steps == -1:
                use_3d_mode = False
            else:
                use_3d_mode = (i < 0 / 10 * len(timesteps)) or (((len(timesteps) - 1 - i) % use_3d_mode_every_m_steps) == 0) 

            t = t[None].repeat(B)
            
            latents_noisy, result = self.inference_one_step(cameras, latents_noisy, text_embeddings, uncond_text_embeddings, t, guidance_scale=guidance_scale, use_3d_mode=use_3d_mode)
            results.append(result)

        if num_refine_steps > 0:
            assert 'gaussians' in results[-1]
            assert self.refiner is not None
            assert dense_cameras is not None
            gaussians = self.refiner.refine_gaussians(results[-1]['gaussians'], text, dense_cameras=dense_cameras)
            images_pred, _, _, _, _ = self.model.render(cameras, gaussians, h=self.image_size, w=self.image_size)
            result = {"gaussians": gaussians, "images_pred": images_pred}
            results.append(result)

        return results 
    # @torch.cuda.amp.autocast(enabled=True)
    # @torch.no_grad()
    # def inference(self, cameras, text, ref_camera=None, latents=None, task='text_to_3d', num_inference_steps=100, num_refine_steps=0, guidance_scale=7.5, use_3d_mode_every_m_steps=10, negative_text="ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate", is_scene=False):
    #     B, N = cameras.shape[:2]
    #     self.scheduler.set_timesteps(num_inference_steps, self.device)
    #     timesteps = self.scheduler.timesteps

    #     latents_noisy = torch.randn(B, N, self.latent_channel, self.latent_size, self.latent_size, device=self.device)

    #     # text_embeddings = self.model.encode_text([(text + ' 3D scene.') if is_scene else text])
    #     text_embeddings = self.model.encode_text([text])
    #     uncond_text_embeddings =  self.model.encode_text([negative_text]).repeat(B, 1, 1)

    #     assert use_3d_mode_every_m_steps != 1, "use_3d_mode_every_m_steps can not be 1"

    #     if use_3d_mode_every_m_steps == -1:
    #         guidance_scale = guidance_scale
    #     else:
    #         guidance_scale = guidance_scale * use_3d_mode_every_m_steps / (use_3d_mode_every_m_steps - 1)

    #     results = []
    #     for i, t in tqdm.tqdm(enumerate(timesteps), total=len(timesteps)):
    #         if use_3d_mode_every_m_steps == -1:
    #             use_3d_mode = False
    #         else:
    #             use_3d_mode = (i < 0 / 10 * len(timesteps)) or (((len(timesteps) - 1 - i) % use_3d_mode_every_m_steps) == 0) 

    #         t = t[None].repeat(B)
            
    #         latents_noisy, result = self.inference_one_step(cameras, latents_noisy, text_embeddings, uncond_text_embeddings, t, latents=latents, task=task, guidance_scale=guidance_scale, use_3d_mode=use_3d_mode)
    #         results.append(result)

    #     if num_refine_steps > 0:
    #         assert 'gaussians' in results[-1]
    #         assert self.refiner is not None
    #         assert task == 'text_to_3d'

    #         gaussians = self.refiner.refine_gaussians(results[-1]['gaussians'], text, ref_camera=ref_camera)
    #         images_pred, _, _, _, _ = self.model.render(cameras, gaussians, h=self.image_size, w=self.image_size)
    #         result = {"gaussians": gaussians, "images_pred": images_pred}
    #         results.append(result)

    #     return results 

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    args, extras = parser.parse_known_args()

    try:
        num_nodes = int(os.environ["NUM_NODES"])
    except:
        os.environ["NUM_NODES"] = '1'
        num_nodes = 1
    
    print(num_nodes)

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    train_dataset = import_str(opt['dataset']['module'])(**opt['dataset']['args'], fake_length=250 * opt.training.batch_size * len(opt.training.gpus) * opt.training.accumulate_grad_batches * num_nodes)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.training.batch_size, num_workers=opt.training.num_workers, shuffle=False)
    
    val_dataset = import_str(opt['dataset']['module'])(**opt['dataset']['args'], fake_length=4 * len(opt.training.gpus) * num_nodes)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=1, shuffle=False)

    system = Dual3DSystem(opt)

    trainer = Trainer(
            default_root_dir=f"logs/{opt.experiment_name}",
            max_steps=opt.training.max_steps,
            check_val_every_n_epoch=opt.training.check_val_every_n_epoch,
            log_every_n_steps=1,
            accumulate_grad_batches=opt.training.accumulate_grad_batches,
            precision=opt.training.precision,
            accelerator='gpu',
            gpus=opt.training.gpus,
            strategy=DDPPlugin(find_unused_parameters=False)
                               if len(opt.training.gpus) > 1 else None,
            benchmark=True,
            gradient_clip_val=opt.training.gradient_clip_val,
            resume_from_checkpoint=opt.training.resume_from_checkpoint,
            # track_grad_norm=1,
            # detect_anomaly=True,
            num_nodes=num_nodes,
        )
    
    trainer.fit(model=system, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        