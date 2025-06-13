"""
Trainer of GS-LDM System
"""
#pyling:disable=import-error
import copy
import random
import tqdm
import lpips
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
from diffusers import DDIMScheduler
from lightning import LightningModule
import einops
from einops import rearrange

from prometheus.models import GMLDMModel
from prometheus.utils.image_utils import postprocess_image, colorize_depth_maps

class GMLDMSystem(LightningModule):
    '''GMLDM system'''
    def __init__(self, opt, mode = 'training'):
        super().__init__()
        self.save_hyperparameters(opt)
        self.opt = opt

        self.image_size = self.opt.network.image_size
        self.latent_size = self.opt.network.latent_size
        self.latent_channel = self.opt.network.latent_channel

        self.model = GMLDMModel(opt)
        # if mode == 'training' and opt.training.resume_from_director3d:
        #     director3d_ckpt = torch.load(opt.training.resume_from_director3d)['gm_ldm']
        #     self.model.load_state_dict(director3d_ckpt, strict=False)
        #     print(f"Resume GM-LDM form pretrained Director 3D ckpt {opt.training.resume_from_director3d}. ")

        

        # if mode == 'training':
        #     self.disp_fn = AutoModelForDepthEstimation.from_pretrained(opt.training.depth_model_path) # works well
        #     #self.disp_fn = pipeline(task="depth-estimation", model="pretrained/huggingface/depth-anything/Depth-Anything-V2-Small-hf").model.eval() # doesn't work
        #     self.register_buffer("disp_image_mean", torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), persistent=False)
        #     self.register_buffer("disp_image_std", torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), persistent=False)
        #     #import ipdb; ipdb.set_trace()
        #     self.lpips_fn = lpips.LPIPS(net='vgg', version='0.1', model_path=opt.training.lpips_model_path).eval().requires_grad_(False)
        #     self.model_ema = copy.deepcopy(self.model).requires_grad_(False)

        # self.num_input_views = self.opt.training.num_input_views
        # self.num_novel_views = self.opt.training.num_novel_views

        self.scheduler = DDIMScheduler(beta_schedule='scaled_linear', beta_start=0.00085, beta_end=0.012, prediction_type="sample", clip_sample=False, steps_offset=9, rescale_betas_zero_snr=True, set_alpha_to_one=True)

        self.register_buffer("alphas_cumprod", self.scheduler.alphas_cumprod, persistent=False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = 0
        self.max_step = int(self.num_train_timesteps)


        # self.refiner = None

    def load_from_director3d_ckpts(self):
            director3d_ckpt = torch.load(self.opt.training.resume_from_director3d)['gm_ldm']
            self.model.load_state_dict(director3d_ckpt, strict=False)
            print(f"Resume GM-LDM form pretrained Director 3D ckpt {self.opt.training.resume_from_director3d}. ")
    
    def configure_noise_scheduler(self):
        """_summary_

        _extended_summary_
        """
        self.scheduler = DDIMScheduler(beta_schedule='scaled_linear', beta_start=0.00085, beta_end=0.012, prediction_type="sample", clip_sample=False, steps_offset=9, rescale_betas_zero_snr=True, set_alpha_to_one=True)
        self.register_buffer("alphas_cumprod", self.scheduler.alphas_cumprod, persistent=False)
    
    def configure_optimizers(self):
        """"""
        params = []
        for p in self.model.parameters():
            if p.requires_grad: params.append(p)
        optimizer = torch.optim.AdamW(params, lr=self.opt.training.learning_rate / self.opt.training.accumulate_grad_batches, weight_decay=self.opt.training.weight_decay, betas=self.opt.training.betas)
        return optimizer

    @torch.amp.autocast("cuda", enabled=False)
    def prepare_data_for_different_task(self, latents_noisy, texts, cameras, t, input_latents, task='text_to_3d'):
        """"""
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

    @torch.no_grad()
    def get_depth_gt(self, x, return_disp = False):
        B, N, C, H, W = x.shape
        x = x.flatten(0, 1)
        #x = F.interpolate(x, size=(518, 518), align_corners=False, mode='bicubic')
        inputs = dict(pixel_values =((x + 1)/2 - self.disp_image_mean) / self.disp_image_std)
        disps = self.disp_fn(**inputs).predicted_depth.unsqueeze(1)
        disps = F.interpolate(disps, size=(H, W), align_corners=False, mode='bilinear')
        disps_flatten = disps.flatten(1, -1)
        min_disps_flatten = disps_flatten.min(dim=1)[0].reshape(B * N, 1, 1, 1)
        max_disps_flatten = disps_flatten.max(dim=1)[0].reshape(B * N, 1, 1, 1)
        disps = (disps - min_disps_flatten) / (max_disps_flatten - min_disps_flatten + 1e-5)

        if False:
            colorize_depth = colorize_depth_maps(1 / (disps + 1e-2), 0, 1, "Spectral") # Disparity Visualization
            colorize_depth_vis = einops.rearrange(colorize_depth, 'B C H W -> H (B W) C').cpu()
            image_vis =  einops.rearrange((x + 1)/2 , 'B C H W -> H (B W) C').cpu()
            image_vis = torch.cat((colorize_depth_vis, image_vis), dim = 0).clip(0,1).numpy()
            plt.imsave('depth_pred_vis.png',image_vis)

        disps = disps.unflatten(0, (B, N))
        if return_disp:
            return disps
        else:
            return 1 / (disps + 1e-2)

    def add_noise(self, x, noise, t):
        """BB"""
        x_noisy = self.scheduler.add_noise(x, noise, t)
        return x_noisy

    def training_step(self, batch, _):
        self.lpips_fn.eval().requires_grad_(False)
        self.disp_fn.eval().requires_grad_(False)
        self.model.vae.encoder.eval().requires_grad_(False)
        self.model.unet.train()
        self.model.vae.decoder.train()
        update_average(self.model_ema, self.model)

        images_original, cameras, texts, images2d, _texts2d = (
            batch['images_mv'], batch['cameras_mv'], batch['text_mv'], batch['image'], batch['text'])

        loss_total = 0
        # is_2d = (not self.global_step % 2)
        # #2d part:n b
        # if is_2d:
        images2d = images2d.flatten(0, 1).unsqueeze(1)
        texts2d = []
        for text2d in _texts2d:
            texts2d += text2d
        B, _, C, H, W = images2d.shape

        with torch.no_grad():
            input_latents2d = self.model.encode_image(images2d)
            #depths2d = self.get_depth_gt(images2d, return_disp=True)

        t = torch.randint(0, self.num_train_timesteps, (B,), dtype=torch.long, device=self.device)
        latents_noisy = self.add_noise(input_latents2d, torch.randn_like(input_latents2d), t)
        t = t.unsqueeze(1)
        text_embeddings = self.model.encode_text(texts2d)
        random_cameras = cameras[:, :1].repeat(B//cameras.shape[0], 1, 1)
        latents_pred, gaussians2d = self.model.denoise(latents_noisy, text_embeddings, t, random_cameras, return_3d=True)
        #gaussians2d: [8, 256*256, 1]

        # Image denoising loss
        loss_sv_latent_mse = F.mse_loss(input_latents2d, latents_pred)
        loss_total += loss_sv_latent_mse * self.opt.losses.lambda_sv_latent_mse
        self.log('losses/single_view/loss_sv_latent_mse', loss_sv_latent_mse, sync_dist=True)
        # # Depth denoising loss (as in Marigold and ChronoDepth)
        # loss_sv_latent_depth = F.mse_loss(input_latents2d, latents_pred)
        # loss_total += loss_sv_latent_mse * self.opt.losses.lambda_sv_latent_mse
        # self.log('losses/single_view/loss_sv_latent_mse', loss_sv_latent_mse, sync_dist=True)
        with torch.amp.autocast("cuda", enabled=False):
            images2d_pred, depths2d_pred, _, reg_losses2d, _ = self.model.render(random_cameras, gaussians2d, h=self.image_size, w=self.image_size)
            # Rendering RGB loss
            images2d = images2d.reshape(B, -1, self.image_size, self.image_size)
            images2d_pred = images2d_pred.reshape(B, -1, self.image_size, self.image_size)

            loss_sv_image_mse = F.mse_loss(images2d, images2d_pred)
            loss_sv_image_lpips = self.lpips_fn(images2d, images2d_pred).mean()
            loss_total += loss_sv_image_mse * self.opt.losses.lambda_sv_image_mse
            loss_total += loss_sv_image_lpips * self.opt.losses.lambda_sv_image_lpips
            # if True: #for depth loss debug
            #     loss_total += depths2d_pred.mean() * self.opt.losses.lambda_mv_image_lpips

        self.log('losses/single_view/loss_image_mse', loss_sv_image_mse, sync_dist=True)
        self.log('losses/single_view/loss_image_lpips', loss_sv_image_lpips, sync_dist=True)

        # Rendering Depth loss
        # depths2d = depths2d.reshape(B, self.image_size * self.image_size)
        # depths2d_pred = depths2d_pred.reshape(B, self.image_size * self.image_size)
        # loss_sv_depth = depth_loss(depths2d / 100, depths2d_pred / 100)
        # loss_total += loss_sv_depth * self.opt.losses.lambda_sv_depth
        # self.log('losses/single_view/loss_image_depth', loss_sv_depth, sync_dist=True)

        # Multi-view part
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
        loss_total += loss_mv_latent_mse * self.opt.losses.lambda_mv_latent_mse

        # novel view synthesis
        with torch.amp.autocast("cuda", enabled=False):
            nv_images = images_original[:, self.num_input_views:]
            nv_images_pred, depths_pred, masks, reg_losses, states = self.model.render(cameras[:, self.num_input_views:], gaussians, h=self.image_size, w=self.image_size)

            nv_images = nv_images.reshape(B * self.num_novel_views, -1, self.image_size, self.image_size)
            nv_images_pred = nv_images_pred.reshape(B * self.num_novel_views, -1, self.image_size, self.image_size)

            loss_mv_image_mse = F.mse_loss(nv_images_pred, nv_images)
            loss_mv_image_lpips = self.lpips_fn(nv_images_pred, nv_images).mean()

            loss_total += loss_mv_image_mse * self.opt.losses.lambda_mv_image_mse
            loss_total += loss_mv_image_lpips * self.opt.losses.lambda_mv_image_lpips
            # if True: #for depth loss debug
            #     loss_total += depths_pred.mean() * self.opt.losses.lambda_mv_image_lpips

            # loss_total += reg_losses['loss_reg_opacity'] * self.opt.losses.lambda_reg_opacity
            # loss_total += reg_losses['loss_reg_scales'] * self.opt.losses.lambda_reg_scales
        self.log('losses/3d/loss_mv_latent_mse', loss_mv_latent_mse, sync_dist=True)
        self.log('losses/3d/loss_mv_image_mse', loss_mv_image_mse, sync_dist=True)
        self.log('losses/3d/loss_mv_image_lpips', loss_mv_image_lpips, sync_dist=True)

            # for key, value in reg_losses.items():
            #     self.log(f'losses/reg/{key}', value, sync_dist=True)
            # for key, value in states.items():
            #     self.log(f'states/{key}', value, sync_dist=True)
        # if False:
        #     scaler = torch.cuda.amp.GradScaler()
        #     loss_mv_image_mse_ = scaler.scale(loss_mv_image_mse)
        #     grad_params = torch.autograd.grad(outputs=loss_mv_image_mse_,
        #                                   inputs=self.model.vae.decoder.parameters(),
        #                                   create_graph=True)
        #     with torch.autograd.detect_anomaly():
        #         loss_mv_image_lpips_ = scaler.scale(loss_mv_image_lpips)
        #         grad_params = torch.autograd.grad(outputs=loss_mv_image_lpips_,
        #                                     inputs=self.model.vae.decoder.parameters(),
        #                                     create_graph=True)
        return loss_total

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """XX"""

        images_original, cameras, texts, images2d, _texts2d  = (
            batch['images_mv'], batch['cameras_mv'], batch['text_mv'], batch['image'], batch['text'])

        # 2d part:
        #rearrange(images2d, "B N C H W")
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
        depths2d = self.get_depth_gt(images2d, return_disp=True) # [B 1 1 H W]

        with torch.amp.autocast("cuda", enabled=False):
            images2d_pred, depths2d_pred, _, _, _ = self.model.render(random_cameras, gaussians2d, h=self.image_size, w=self.image_size)

        self.logger.log_image(f"val/single_view_gt", postprocess_image(images2d.flatten(0, 1), return_PIL = True), self.global_step)
        self.logger.log_image(f"val/single_view",  postprocess_image(images2d_pred.flatten(0, 1), return_PIL = True), self.global_step)
        # Log and visualize depth (single view)
        depths2d = rearrange(depths2d, 'B N C H W -> (B N C) H W')
        depths2d_pred = rearrange(depths2d_pred,'B N C H W -> (B N C) H W')
        self.logger.log_image("val/single_view_depth_puesudo_gt",  postprocess_image(colorize_depth_maps(depths2d, 0, 1, "Spectral_r"), 0, 1,return_PIL = True), self.global_step)
        #self.logger.log_image(f"val/single_view_depth_pred/{int(batch_idx)}",  postprocess_image(colorize_depth_maps(1/(depths2d_pred+1e-2), None, None, "Spectral_r"), 0, 1, return_PIL = True), self.global_step) # convert to disparity and visualiza
        self.logger.log_image("val/single_view_depth_pred",  postprocess_image(colorize_depth_maps(depths2d_pred, None, None, "Spectral"), 0, 1, return_PIL = True), self.global_step)# directly visualiza metric depth

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
            nv_images_pred, nv_depth_pred, _, _, _ = self.model.render(cameras[:, self.num_input_views:], gaussians, h=self.image_size, w=self.image_size)

        self.logger.log_image("val/input_views", postprocess_image(input_views.flatten(0, 1), return_PIL=True), self.global_step)
        self.logger.log_image("val/novel_view_gt", postprocess_image(nv_images.flatten(0, 1), return_PIL=True), self.global_step)
        self.logger.log_image("val/novel_view", postprocess_image(nv_images_pred.flatten(0, 1), return_PIL=True), self.global_step)
        # Log and visualize depth (multiview)
        nv_depth_gt = self.get_depth_gt(input_views, return_disp=True) # [1 N 1 H W]
        nv_depth_gt = rearrange(nv_depth_gt, 'B N C H W -> (B N C) H W')
        nv_depth_pred = rearrange(nv_depth_pred,'B N C H W -> (B N C) H W')
        self.logger.log_image("val/novel_view_depth_puesudo_gt",  postprocess_image(colorize_depth_maps(nv_depth_gt, 0, 1, "Spectral_r"), 0, 1,return_PIL = True), self.global_step)
        # self.logger.log_image(f"val/novel_view_depth/{int(batch_idx)}",  postprocess_image(colorize_depth_maps(1/(nv_depth_pred+1e-2), None, None, "Spectral_r"), 0, 1, return_PIL = True), self.global_step) # convert to disparity and visualiza
        self.logger.log_image("val/novel_view_depth",  postprocess_image(colorize_depth_maps(nv_depth_pred, None, None, "Spectral"), 0, 1, return_PIL = True), self.global_step) # directly visualiza metric depth

        return

    def inference_one_step(self, cameras, latents_noisy, text_embeddings, uncond_text_embeddings, t, guidance_scale=10, use_3d_mode=True):
        """"""
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

            return latents_less_noisy, {"gaussians": gaussians, "images_pred": images_pred}
            
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

        return latents_less_noisy, {'latents_pred': _latents_pred}

    @torch.amp.autocast("cuda", enabled=True)
    @torch.no_grad()
    def inference(self, cameras, text, dense_cameras=None, refiner=None, num_inference_steps=100, guidance_scale=7.5, use_3d_mode_every_m_steps=10, negative_text=""):
        B, N = cameras.shape[:2]
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = self.scheduler.timesteps

        latents_noisy = torch.randn(B, N, self.latent_channel, self.latent_size, self.latent_size, device=self.device) # self.latent_channel == 4 instead of 512

        text_embeddings = self.model.encode_text([text])
        uncond_text_embeddings =  self.model.encode_text([negative_text]).repeat(B, 1, 1)

        assert use_3d_mode_every_m_steps != 1, "use_3d_mode_every_m_steps can not be 1"

        if use_3d_mode_every_m_steps == -1:
            guidance_scale = guidance_scale
        else:
            guidance_scale = guidance_scale * use_3d_mode_every_m_steps / (use_3d_mode_every_m_steps - 1)

        for i, t in tqdm.tqdm(enumerate(timesteps), total=len(timesteps), desc='Denoising image sequence...'):
            if use_3d_mode_every_m_steps == -1:
                use_3d_mode = False
            else:
                use_3d_mode = not ((len(timesteps) - 1 - i) % use_3d_mode_every_m_steps)

            t = t[None].repeat(B)
            
            latents_noisy, result = self.inference_one_step(cameras, latents_noisy, text_embeddings, uncond_text_embeddings, t, guidance_scale=guidance_scale, use_3d_mode=use_3d_mode)

        if refiner is not None:
            assert 'gaussians' in result
            assert dense_cameras is not None
            gaussians = refiner.refine_gaussians(result['gaussians'], text, dense_cameras=dense_cameras)
            images_pred, _, _, _, _ = self.model.render(cameras, gaussians, h=self.image_size, w=self.image_size)
            result = {"gaussians": gaussians, "images_pred": images_pred}

        return result 

def update_average(model_tgt, model_src, beta=0.995):
    """"""
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

