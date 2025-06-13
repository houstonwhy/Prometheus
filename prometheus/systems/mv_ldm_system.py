"""
Trainer of Multi-view diffusion (text/view cond)
"""
#pyling:disable=import-error
import copy
import random
import tqdm
import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import einops
from easydict import EasyDict
from einops import rearrange, repeat
from prometheus.utils import sample_rays, embed_rays
from prometheus.utils.image_utils import postprocess_image, colorize_depth_maps
from prometheus.utils import import_str
# from torchmetrics.image import FrechetInceptionDistance as FID
from .depth_loss import depth_to_disp
from .base_system import BaseSystem
class MVLDMSystem(BaseSystem):
    '''Multi-view LDM system'''
    def __init__(self, opt, mode = 'training'):
        super().__init__(opt, mode)
        self.num_ref_views = opt.training.num_ref_views
        self.num_pred_views = opt.training.num_pred_views
        assert (self.num_ref_views + self.num_pred_views) == (self.num_input_views + self.num_novel_views)
        self.ajust_sigma_on_N = False
        self.noise_type = opt.training.get('noise_type', 'vanilla')
        self.task_type = opt.training.get('task_type', 'text_to_3d')
        #noise_type: 'vanilla', 'view_cond', 
        if self.noise_type == 'view_cond':
            self.random_ref_num = opt.training.random_ref_num
            assert self.model.extra_latent_channel == 11

        # Resume pretrained mvldm and gsdecoder (might has different shape in some layers)
        mvldm_path = opt.training.get('mvldm_path', None)
        if mvldm_path and mode == 'training' and (self.existing_checkpoint_path is None):
            try:
                self.load_from_mvldm_ckpt(mvldm_path)
            except:
                print(f"Fail to load MV-LDM state dict from pretrained ckpt {mvldm_path}. ")
        
        # Init & resume gsdecoder for rendering loss
        use_gsdecoder = opt.training.get('use_gsdecoder', False)
        if use_gsdecoder :
            assert hasattr(self.opt, 'gsdecoder') and  hasattr(self.opt.training, 'gsdecoder_path')
            self.gsdecoder = import_str(self.opt.gsdecoder.module)(opt.gsdecoder.network, use_ema_norm = False, mode = 'inference',)
            gsdecoder_ckpt_path = self.opt.training.gsdecoder_path
            # try:
            if mode == 'training' and (gsdecoder_ckpt_path is not None):
                self.load_from_gsdecoder_ckpt(gsdecoder_ckpt_path)
        self.rendering_batch_size = opt.training.get('rendering_batch_size', 1)
        self.use_gsdecoder = opt.training.get('use_gsdecoder', False)
        self.tune_decoder_only = opt.training.get('tune_decoder_only', False)


    def load_from_mvldm_ckpt(self, mvldm_ckpt_path):
        """Load legacy mvldm ckpts that might has shape difference"""
        mvldm_ckpt = torch.load(mvldm_ckpt_path, weights_only=False, map_location='cpu')['state_dict']
        # Load GS-Decoder
        mvldm_state_dict = {}
        for key, value in mvldm_ckpt.items():
            if 'model.unet' in key:
                mvldm_state_dict[key.replace('model.unet.', '')] = value
        try:
            self.model.unet.load_state_dict(mvldm_state_dict)
        except:
            self.model.unet.state_dict()['input_blocks.0.0.weight'][:,:14]=  mvldm_state_dict['input_blocks.0.0.weight']
            self.model.unet.state_dict()['out.2.bias'][:14] = mvldm_state_dict['out.2.bias']
            self.model.unet.state_dict()['out.2.weight'][:14] = mvldm_state_dict['out.2.weight']
            
            current_model_state_dict = self.model.unet.state_dict()
            keys_to_delete = [key for key, param in mvldm_state_dict.items() 
                if key not in current_model_state_dict or 
                param.size() != current_model_state_dict[key].size()]
            for key in keys_to_delete:
                del mvldm_state_dict[key]
            self.model.unet.load_state_dict(mvldm_state_dict, strict=False)
        
        del mvldm_ckpt
        print(f"Resume MV-LDM form pretrained ckpt {mvldm_ckpt_path}. ")

    def load_from_gsdecoder_ckpt(self, gsdecoder_ckpt_path):
        """Load pretrained gsdecoder ckpt"""
        gsdecoder_ckpt = torch.load(gsdecoder_ckpt_path, weights_only=False, map_location='cpu')['state_dict']
        # Load GS-Decoder
        gsdecoder_state_dict = {}
        for key, value in gsdecoder_ckpt.items():
            if ('model_ema.' in key) and ('lpips_fn' not in key):
                gsdecoder_state_dict[key.replace('model_ema.', '')] = value
        try:
            self.gsdecoder.load_state_dict(gsdecoder_state_dict,strict=True)
        except:
            replace_after_emanorm = [
                ("vae.decoder.up.1.upsample.conv.weight", "vae.decoder.up.1.upsample.conv.0.weight"),
                ("vae.decoder.up.1.upsample.conv.bias", "vae.decoder.up.1.upsample.conv.0.bias"),
                ("vae.decoder.up.2.upsample.conv.weight", "vae.decoder.up.2.upsample.conv.0.weight"),
                ("vae.decoder.up.2.upsample.conv.bias", "vae.decoder.up.2.upsample.conv.0.bias"),
                ("vae.decoder.up.3.upsample.conv.weight", "vae.decoder.up.3.upsample.conv.0.weight"),
                ("vae.decoder.up.3.upsample.conv.bias","vae.decoder.up.3.upsample.conv.0.bias")]
            
            for oo, tt in replace_after_emanorm:
                gsdecoder_state_dict[tt] = gsdecoder_state_dict[oo]
                del gsdecoder_state_dict[oo]
            self.gsdecoder.load_state_dict(gsdecoder_state_dict,strict=False)
        del gsdecoder_ckpt 
        # del gsdecoder_state_dict
        print(f"Resume GS-Decoder form pretrained ckpt {gsdecoder_ckpt_path}. ")

    # @torch.amp.autocast("cuda", enabled=True) # -> fp16 by default!!!!
    def training_step(self, batch):
        self.lpips_fn.eval().requires_grad_(False)
        self.disp_fn.eval().requires_grad_(False)
        self.model.vae.eval().requires_grad_(False)
        if hasattr(self, 'gsdecoder'):
            if self.opt.losses.get('lambda_gs_image_mse', 0) > 0:
                self.gsdecoder.train()
                self.gsdecoder.requires_grad_(True)
            else:
                self.gsdecoder.eval().requires_grad_(False)
        # self.model.vae.eval().requires_grad_(False)
        if self.tune_decoder_only:
            self.model.unet.eval().requires_grad_(False)
        else:
            self.model.unet.train()
            self.update_average(self.model_ema, self.model)
        log_dict = {}
        loss_total = 0
        # Multi-view part
        if 'images_mv' in  batch.keys():
            loss_, log_dict_= self.forward_multi_view(batch=batch, mode='train')
            loss_total += loss_
            log_dict.update(log_dict_)
        # Single-view part
        if 'image' in batch.keys() and self.task_type == 'text_to_3d':
            loss_, log_dict_ = self.forward_single_view(batch=batch, mode='train')
            loss_total += loss_
            log_dict.update(log_dict_)

        log_dict.update({'train/loss_total':loss_total})
        self.log_dict(log_dict, rank_zero_only=True)
        return loss_total
    

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """XX"""
        loss_total = 0.
        log_dict = {}
        #Multiview part
        if 'images_mv' in  batch.keys():
            loss_, log_dict_ = self.forward_multi_view(batch=batch, mode = 'val', batch_idx=batch_idx)
            loss_total += loss_
            log_dict.update(log_dict_)
        # 2d part:
        #rearrange(images_sv, "B N C H W")
        if 'image' in  batch.keys():
            loss_, log_dict_ = self.forward_single_view(batch=batch, mode = 'val', batch_idx=batch_idx)
            loss_total += loss_
            log_dict.update(log_dict_)

        log_dict.update({'val/loss_total':loss_total})
        self.log_dict(log_dict, rank_zero_only=True)
        return

    def forward_single_view(self, batch, mode = 'train', batch_idx = 0):
        """Training on single view dattasets e.g. LAION5B, SA1B, JourneyDB"""
        loss_total = 0.
        log_dict = {}
        images_sv, _texts_sv  = batch['image'], batch['text']
        images_sv = images_sv.flatten(0, 1).unsqueeze(1)
        texts_sv = []
        #BUG pytorch tackle list in a wired manner 
        for text_sv in zip(*_texts_sv):
             texts_sv += text_sv
        #[(0, 2, 4, 6), (1, 3, 5, 7)] -> [0,1,2,3,4,5,6,7]
        # for text_sv in zip(_texts_sv):
        #     texts_sv += text_sv
        B, N, C, H, W = images_sv.shape
        #with torch.amp.autocast("cuda", enabled=False):
        depths_sv = self.get_depth_gt(images_sv, return_disp=True, normalize=True) # [B 1 1 H W]
        depths_sv_in = repeat(depths_sv * 2 - 1, "B N C H W -> B N (C c3) H W", c3 = 3)
        input_latents = self.model.encode_image(images_sv)
        input_latents_depth = self.model.encode_image(depths_sv_in)

        #text_embeddings = self.model.encode_text(texts_sv)
        if not 'cameras_mv' in batch.keys():
            cameras = torch.tensor([1,0,0,0,
                                    0,1,0,0,
                                    0,0,1,0,
                                    207.3, 207.3, 
                                    128.0000, 128.0000, 
                                    256.0000, 256.0000]).to(images_sv)
            cameras = repeat(cameras, 'C -> B N C', B = B, N = N)
        else:
            cameras =  batch['cameras_mv']
            cameras = cameras[:, 1:2].repeat(B//cameras.shape[0], 1, 1)

        if 'RGBD' in self.decoder_mode:
            input_latents = torch.cat((input_latents, input_latents_depth), dim = 2) # [B N 8 H W] [-1,1]
        Hl, Wl = input_latents.shape[3], input_latents.shape[4]

        if mode == 'train':
            #TODO random drop
            texts_sv, cameras = self.prepare_data_for_different_task(texts_sv, cameras, task='text_to_3d')

        if True:
            text_embeddings = self.model.encode_text(texts_sv)
        else:
            pass
        
        #TODO prepare & drop raymap
        
        if self.scheduler_type == 'ddim' :
            # sample t and denoise
            t = torch.randint(0, self.num_train_timesteps, (B,), dtype=torch.long, device=self.device)
            latents_noisy = self.add_noise(input_latents, torch.randn_like(input_latents), t)
            t = t.unsqueeze(1)
            
            #random_cameras = cameras[:, :1].repeat(B//cameras.shape[0], 1, 1)
            latents_pred = self.model.denoise(latents_noisy, text_embeddings, t, cameras, raymap_mode = 'lowres')
            
            #weight
            # Image denoising loss
            loss_sv_latent_mse = F.mse_loss(latents_pred[:,:,0:4], input_latents[:,:,0:4], )
            loss_sv_latent_mse_depth = F.mse_loss(latents_pred[:,:,4:8], input_latents[:,:,4:8])
        
        elif (self.scheduler_type == 'edm') or \
            (self.scheduler_type  ==  'vprediction'):
            #-------------------for edm------------------------------------
            # preprese noisy latents & sigma
            # if self.high_noise_level:
            #     p_std, p_mean, sigma_data = 2.0, 1.5, 1.0
            # else:
            #     p_std, p_mean, sigma_data = 1.2, -1.2, 0.5
            noise = torch.randn_like(input_latents)
            if self.scheduler_type == 'edm':
                #BUG we need lower noise level for single view 2.0, 1.5 -> 1.2,-0.5
                p_std, p_mean = 1.2, -0.5 # same as stable video diffusion
                # p_std, p_mean =self.scheduler_vars.p_std, self.scheduler_vars.p_mean

                rnd_normal = torch.randn([noise.shape[0], 1, 1, 1, 1], device=input_latents.device).repeat((1,N,1,1,1))
                sigmas = (rnd_normal * p_std +  p_mean).exp()
                t_cond = c_noise(sigmas).view(B,N)
            else:
                t = torch.randint(0, self.num_train_timesteps, (B,), dtype=torch.long, device=noise.device)
                # Wrong: sigmas = repeat(self.sigmas[t.cpu()].to(noise.device),'B -> B N C H W', N = N, C = 1, H = 1, W = 1)
                sigmas = repeat(self.timesteps_to_sigmas[t.cpu()],'B -> B N C H W', N = N, C = 1, H = 1, W = 1).to(noise.device)
                t_cond = repeat(self.timesteps_to_tconds[t.cpu()], 'B -> B N', N = N).to(noise.device)

            if self.ajust_sigma_on_N: # Ajust noise for single view
                sigmas = sigmas / math.sqrt(self.num_pred_views)
            
            c_skip = self.scheduler_vars.c_skip
            c_out = self.scheduler_vars.c_out
            c_in = self.scheduler_vars.c_in
            c_noise = self.scheduler_vars.c_noise
            weight = self.scheduler_vars.weight
            # input_sigmas = repeat(c_noise(sigmas),'B -> B N', N = N)

            # sigmas = sigmas[:, None, None, None, None]
            noisy_latents = input_latents + noise * sigmas
            noisy_input_latents = c_in(sigmas) * noisy_latents
            # preprese cond latents & cond sigma (raymap)
            rays_o, rays_d = sample_rays(cameras.flatten(0, 1), h=Hl, w=Wl, N=-1)# [B, N, H*W, 3]
            rays_embeddings = rearrange(embed_rays(rays_o, rays_d), "(B N) (H W) C -> B N C H W", B = B, H = Hl, W = Wl).contiguous().to(noisy_latents.device)
            #Random drop cond_view & raymap with 0.1 prob for CFG
            
            for bb in range(B):
                # if random.random() < self.opt.training.drop_pose_p:
                if random.random() < self.opt.training.drop_pose_p or (texts_sv == ''):
                    # TODO random drop cond image (follow CAT3D)
                    rays_embeddings[bb] = torch.zeros_like(rays_embeddings[bb])

            if self.noise_type in ['view_cond']: # add a little noise on raymap cond
                cond_latents = torch.cat([rays_embeddings, torch.zeros_like(rays_embeddings[:,:,0:1])], 2)
            else:
                cond_latents = rays_embeddings
            
            #cond_latents = self.tensor_to_vae_latent(rgb_batch)
            noisy_input_latents = torch.cat([noisy_input_latents, cond_latents], 2)

            model_preds = self.model.denoise(
                noisy_input_latents, 
                text_embeddings = text_embeddings, 
                t = t_cond
                )
            
            # D_x = c_skip * x + c_out * F_x.to
            latents_pred =  c_skip(sigmas) * noisy_latents + c_out(sigmas) * model_preds# adaptive esplion preiction

            loss_sv_latent_mse = (weight(sigmas) * (latents_pred[:,:,0:4] - input_latents[:,:,0:4])**2).mean()
            loss_sv_latent_mse_depth = (weight(sigmas) * (latents_pred[:,:,4:8] - input_latents[:,:,4:8])**2.).mean()
        
        loss_total += loss_sv_latent_mse * self.opt.losses.lambda_sv_latent_mse
        loss_total += loss_sv_latent_mse_depth * self.opt.losses.lambda_sv_latent_mse_depth
 
        log_dict.update({f"{mode}/loss_sv_latent_mse":loss_sv_latent_mse})
        log_dict.update({f"{mode}/loss_sv_latent_mse_depth":loss_sv_latent_mse_depth})

        log_image_flag = (mode == 'val' and batch_idx == 0) or \
            (mode == 'train' and self.global_step % self.log_every_n_step == 0)
        #TODO Novel view rendering loss
        # rendering_loss = self.compute_rendering()
        # prepare gsdecoder input
        if self.use_gsdecoder and self.opt.losses.get('lambda_gs_image_mse', 0) > 0:
            rendering_batch_size = self.rendering_batch_size 
            # gs_in_idx = torch.randperm(N)[:4]
            gs_full_loss, gs_loss_dict, gs_image_dict =  self.rendering_loss_fn(
                latents=latents_pred[:rendering_batch_size*self.single_view_num],
                cameras = cameras[:rendering_batch_size*self.single_view_num],
                # gt images
                images_gt=images_sv[:rendering_batch_size*self.single_view_num],
                depths_gt=depths_sv_in[:rendering_batch_size*self.single_view_num],
                cameras_gt=cameras[:rendering_batch_size*self.single_view_num],
                # images denoise from untt
                log_image = log_image_flag,
                rendering_size=256,
                suffix='svpred_latents',
                view_mode='sv',
                mode=mode)
            
            loss_total += gs_full_loss
            log_dict.update(gs_loss_dict)
        
        # Decode image when eval / snapshot
        if (mode == 'val') or \
            (mode == 'train' and self.global_step % self.log_every_n_step == 0):
            with torch.no_grad():
                images_sv_pred = self.model.decode_latent(latents_pred[:,:,0:4],  mode='image')
                depths_sv_pred = self.model.decode_latent(latents_pred[:,:,4:8],  mode='image').mean(dim = 2, keepdim=True) / 2 + 0.5

            images_sv = rearrange(images_sv, 'B N C H W -> (B N) C H W')
            images_sv_pred = rearrange(images_sv_pred,'B N C H W -> (B N) C H W')
            depths_sv = rearrange(depths_sv, 'B N C H W -> (B N C) H W')
            depths_sv_pred = rearrange(depths_sv_pred,'B N C H W -> (B N C) H W')

            psnr_sv = self.psnr_fn(images_sv, images_sv_pred)
            psnr_sv_depth = self.psnr_fn(depths_sv, depths_sv_pred)

            if mode == 'val':
                log_dict.update({f"{mode}/PSNR_sv":psnr_sv})
                log_dict.update({f"{mode}/PSNR_sv_depth": psnr_sv_depth})

        # Log image when eval / snapshot
        if (mode == 'val' and batch_idx == 0) or \
        (mode == 'train' and self.global_step % self.log_every_n_step == 0):
            # Log Singleview results
            self.logger.log_image(f"snapshot_{mode}/single_view_gt", postprocess_image(images_sv, return_PIL = True), step = self.global_step, caption = texts_sv)
            self.logger.log_image(f"snapshot_{mode}/single_view",  postprocess_image(images_sv_pred, return_PIL = True), step = self.global_step, caption = texts_sv)
            # Log and visualize depth (single view)
            self.logger.log_image(f"snapshot_{mode}/single_view_depth_gt", \
            postprocess_image(colorize_depth_maps(depths_sv, 0, 1, "Spectral_r"), 0, 1, return_PIL = True), step = self.global_step, caption = texts_sv)
            self.logger.log_image(f"snapshot_{mode}/single_view_depth_pred", \
            postprocess_image(colorize_depth_maps(depths_sv_pred, 0, 1, "Spectral_r"), 0, 1, return_PIL = True), step = self.global_step, caption = texts_sv)

            self.logger.log_text(key=f"snapshot_{mode}", 
                                   columns=['single_view_text'], 
                                   data=[t for t in zip(texts_sv)])
            
            inference_results = self.inference(
                cameras=cameras,
                text=texts_sv,
                num_inference_steps=25,
                guidance_scale=7.5,
                get_gs=False
                # use_joint_guidance=True,
            )
            images_sv_full_denoise = postprocess_image(rearrange(inference_results['images_pred'], "B (n1 n2) C H W  -> B C (n1 H) (n2 W)", B = B, n1 = max(1, N // 4), n2 = N // max(1,  N // 4)), return_PIL = True)
            depths_sv_full_denoise = postprocess_image(rearrange(colorize_depth_maps(inference_results['depths_pred'].flatten(0,1), 0, 1,"Spectral_r"), "(B n1 n2) C H W  -> B C (n1 H) (n2 W)", B = B, n1 = max(1, N // 4), n2 = N // max(1,  N // 4)), 0, 1, return_PIL = True)
            self.logger.log_image(f"snapshot_{mode}/single_view_full_denoise", images_sv_full_denoise, self.global_step, caption=texts_sv)
            self.logger.log_image(f"snapshot_{mode}/single_view_full_denoise_depth",depths_sv_full_denoise, self.global_step, caption=texts_sv)

            if self.use_gsdecoder and self.opt.losses.get('lambda_gs_image_mse',0) > 0:
                for k, v in gs_image_dict.items():
                    self.logger.log_image(k, v, self.global_step)
    
    
        return loss_total, log_dict


    def forward_multi_view(self, batch, mode = 'train', batch_idx = 0):
        loss_total = 0.
        log_dict = {}
        images, cameras, texts  = batch['images_mv'], batch['cameras_mv'], batch['text_mv']
        B, N, C, H, W = images.shape

        depths = self.get_depth_gt(images, return_disp=True, normalize=True) # [B 1 1 H W]
        depths_in = repeat(depths * 2 - 1, "B N C H W -> B N (C c3) H W", c3 = 3)
        input_latents = self.model.encode_image(images)
        input_latents_depth = self.model.encode_image(depths_in)
        Hl, Wl = input_latents.shape[3], input_latents.shape[4]
        rays_o, rays_d = sample_rays(cameras.flatten(0, 1), h=Hl, w=Wl, N=-1) # [B, N, H*W, 3]
        if 'RGBD' in self.decoder_mode:
            input_latents = torch.cat((input_latents, input_latents_depth), dim = 2) # [B N 8 H W] [-1,1]

        # prepare ref_idx, target_idx -> only for cat3d style view cond
        if self.noise_type == 'view_cond':
            view_mask = torch.zeros_like(input_latents[:,:,0:1]) 
            # * (N - self.num_ref_views)
            full_idx = torch.randperm(N)
            num_ref_views = random.randint(1, self.num_ref_views) if self.random_ref_num else self.num_ref_views
            # num_ref_views = torch.randint(0, self.num_ref_views)
            ref_idx, target_idx = full_idx[:num_ref_views], full_idx[num_ref_views:]
            ref_idx, target_idx = ref_idx.sort()[0], target_idx.sort()[0]
            view_mask[:,ref_idx] = 1
        elif self.noise_type == 'vanilla':
            target_idx, ref_idx = torch.arange(N, device=input_latents.device), torch.tensor([], dtype=torch.int64, device=input_latents.device)
        else:
            raise ValueError(f'unknown noise type: {self.noise_type}')
        
        # if mode == 'train':
        if self.task_type == 'text_to_3d':
            texts, cameras = self.prepare_data_for_different_task(texts, cameras, task='text_to_3d')
            text_embeddings = self.model.encode_text(texts)
            cond_images = None
        # if self.noise_type == 'view_cond':
        elif self.task_type == 'image_to_3d':
            for bb in range(B):
                if random.random() < self.opt.training.image_to_3d_drop_image_p and mode == 'train':
                    # TODO random drop cond image (follow CAT3D)
                    input_latents[bb,ref_idx] = torch.zeros_like(input_latents[bb,ref_idx])
                    texts[bb] == ''
            cond_images = images[:,ref_idx]
            # Use cond images clip features as text_embeddings
            text_embeddings = self.model.encode_image_clip(cond_images)
        else:
            raise ValueError('?')

        # Image denoising loss
        if self.scheduler_type == 'ddim': #idea from Zero123++
            #TODO rewrite diim into edm formate
            # TODO Drop raymap & text & reference view

            # sample t and denoise for target views
            if self.noise_type == 'vanilla':
                #TODO CAT3D style noise
                t = torch.randint(0, self.num_train_timesteps, (B,), dtype=torch.long, device=self.device)
                t =  repeat(t, "B -> B N", N = N).contiguous()
                t[:,ref_idx] = torch.randint(0, 5, (B,self.num_ref_views), dtype=torch.long, device=self.device) # almost no noise
            elif self.noise_type == 'diffusion_forcing':
                # #TODO futrher, as in DF, add different level of noise for each frame
                # t = torch.randint(0, self.num_train_timesteps, (B,N), dtype=torch.long, device=self.device)
                # t[:,ref_idx] = torch.randint(0, 5, (B,self.num_ref_views), dtype=torch.long, device=self.device)
                # # TODO May be need to add binary mask to denote ref/target frame?
                raise ValueError('unsupport noise type')
            else:
                raise ValueError('unsupport noise type')
            
            latents_noisy = self.add_noise(
                input_latents, 
                torch.randn_like(input_latents), 
                t)
            
            latents_pred = self.model.denoise(latents_noisy, text_embeddings, t, cameras, raymap_mode = 'lowres')

            loss_mv_latent_mse = F.mse_loss(latents_pred[:,target_idx,0:4], input_latents[:,target_idx,0:4])
            loss_mv_latent_mse_depth = F.mse_loss(latents_pred[:,target_idx,4:8], input_latents[:,target_idx,4:8])
        
        elif (self.scheduler_type  == 'edm') or \
            (self.scheduler_type  ==  'vprediction'): #borrow from SVD
            #-------------------for edm------------------------------------
            # preprese noisy latents & sigma
            # if self.high_noise_level:
            #     p_std, p_mean, sigma_data = 2.0, 1.5, 1.0
            # else:
            #     p_std, p_mean, sigma_data = 1.2, -1.2, 0.5
            c_skip = self.scheduler_vars.c_skip
            c_out = self.scheduler_vars.c_out
            c_in = self.scheduler_vars.c_in
            c_noise = self.scheduler_vars.c_noise
            weight = self.scheduler_vars.weight

            noise = torch.randn_like(input_latents)
            if self.scheduler_type == 'edm':
                p_std, p_mean =  self.scheduler_vars.p_std, self.scheduler_vars.p_mean
                # noise = torch.randn_like(input_latents)
                rnd_normal = torch.randn([noise.shape[0], 1, 1, 1, 1], device=images.device).repeat((1,N,1,1,1))
                sigmas = (rnd_normal * p_std + p_mean).exp()
                # input_sigmas = repeat(c_noise(sigmas),'B -> B N', N = N)
                t_cond = c_noise(sigmas).view(B,N)
            else:
                t = torch.randint(0, self.num_train_timesteps, (B,), dtype=torch.long, device=noise.device)
                # Bug here
                # Wrong: sigmas = repeat(self.sigmas[t.cpu()].to(noise.device),'B -> B N C H W', N = N, C = 1, H = 1, W = 1)
                sigmas = repeat(self.timesteps_to_sigmas[t.cpu()],'B -> B N C H W', N = N, C = 1, H = 1, W = 1).to(noise.device)
                # sigmas = sigma_from_t(t)
                t_cond = repeat(self.timesteps_to_tconds[t.cpu()], 'B -> B N', N = N).to(noise.device)

                # if True: # Ajust noise for single view
                #     sigmas = sigmas / math.sqrt(self.num_input_views / (self.num_input_views - len(ref_idx)))

            noisy_latents = input_latents + noise * sigmas
            noisy_input_latents = c_in(sigmas) * noisy_latents
            noisy_input_latents[:,ref_idx] = input_latents[:,ref_idx]

            # preprese cond latents & cond sigma (raymap)
            rays_o, rays_d = sample_rays(cameras.flatten(0, 1), h=Hl, w=Wl, N=-1)# [B, N, H*W, 3]
            rays_embeddings = rearrange(embed_rays(rays_o, rays_d), "(B N) (H W) C -> B N C H W", B = B, H = Hl, W = Wl).contiguous().to(noisy_latents.device)

            for bb in range(B):
                if random.random() < self.opt.training.drop_pose_p or (texts[bb] == ''):
                    # TODO random drop cond image (follow CAT3D)
                    rays_embeddings[bb] = torch.zeros_like(rays_embeddings[bb])

            if self.noise_type == 'view_cond': 
                cond_latents =  torch.cat([rays_embeddings, view_mask], 2)
            else:
                cond_latents = rays_embeddings
            #cond_latents = self.tensor_to_vae_latent(rgb_batch)
            noisy_input_latents = torch.cat([noisy_input_latents, cond_latents], 2)

            model_preds = self.model.denoise(
                noisy_input_latents, 
                text_embeddings = text_embeddings, 
                t = t_cond
                )
            
            # D_x = c_skip * x + c_out * F_x
            latents_pred =  c_skip(sigmas) * noisy_latents + c_out(sigmas) * model_preds # adaptive esplion preiction
            latents_pred[:,ref_idx] = input_latents[:,ref_idx,:8]
            loss_mv_latent_mse = (weight(sigmas) * (latents_pred[:,:,0:4] - input_latents[:,:,0:4])**2)[:,target_idx].mean()
            loss_mv_latent_mse_depth = (weight(sigmas) * (latents_pred[:,:,4:8] - input_latents[:,:,4:8])**2.)[:,target_idx].mean()
        else:
            raise ValueError(f'Unsupport scheculer type{self.scheduler_type}')
    
        #TODO edm loss
        loss_total += loss_mv_latent_mse * self.opt.losses.lambda_mv_latent_mse
        loss_total += loss_mv_latent_mse_depth * self.opt.losses.lambda_mv_latent_mse_depth

        log_dict.update({f"{mode}/loss_mv_latent_mse":loss_mv_latent_mse})
        log_dict.update({f"{mode}/loss_mv_latent_mse_depth": loss_mv_latent_mse_depth})

        log_image_flag = (mode == 'val' and batch_idx == 0) or \
            (mode == 'train' and self.global_step % self.log_every_n_step == 0)
        #TODO Novel view rendering loss
        # rendering_loss = self.compute_rendering()
        # prepare gsdecoder input
        if self.use_gsdecoder and self.opt.losses.get('lambda_gs_image_mse', 0) > 0:
            rendering_batch_size = self.rendering_batch_size 
            # gs_in_idx = torch.randperm(N)[:4]
            _idx = random.sample(range(cameras.shape[1]), k=4)
            # if latents_pred.shape[1] == 8:
            # elif latents_pred.shape[1] == 4:
            #     _idx = [0,1,2,3]
            gs_full_loss, gs_loss_dict, gs_image_dict =  self.rendering_loss_fn(
                latents=latents_pred[:rendering_batch_size,_idx],
                cameras = cameras[:rendering_batch_size,_idx],
                # gt images
                images_gt=images[:rendering_batch_size],
                depths_gt=depths[:rendering_batch_size],
                cameras_gt=cameras[:rendering_batch_size],
                # images denoise from untt
                log_image = log_image_flag,
                rendering_size=256,
                suffix='pred_latents',
                mode=mode)
            
            loss_total += gs_full_loss
            log_dict.update(gs_loss_dict)
            
            if log_image_flag:
                with torch.no_grad():
                    _, _, gs_image_gt_dict =  self.rendering_loss_fn(
                    latents=input_latents[:rendering_batch_size,_idx],
                    cameras = cameras[:rendering_batch_size,_idx],
                    # gt images
                    cameras_gt=cameras[:rendering_batch_size],
                    # images denoise from untt
                    log_image = log_image_flag,
                    suffix='gt_latents',
                    rendering_size=256,
                    mode=mode)

                gs_image_dict.update(gs_image_gt_dict)

           
        # Decode image when eval / snapshot
        if (mode == 'val')  or \
            (mode == 'train' and self.global_step % self.log_every_n_step == 0):
            reorder_idx = torch.cat((target_idx, ref_idx))
            with torch.no_grad():
                images_mv_pred = self.model.decode_latent(latents_pred[:,:,0:4],  mode='image')
                depths_mv_pred = self.model.decode_latent(latents_pred[:,:,4:8],  mode='image').mean(dim = 2, keepdim=True) / 2 + 0.5
            
            images_mv = rearrange(images[:,reorder_idx], 'B N C H W -> (B N) C H W')
            images_mv_pred = rearrange(images_mv_pred[:,reorder_idx],'B N C H W -> (B N) C H W')
            depths_mv = rearrange(depths[:,reorder_idx], 'B N C H W -> (B N C) H W')
            depths_mv_pred = rearrange(depths_mv_pred[:,reorder_idx],'B N C H W -> (B N C) H W')
            # self.fid_fn = 
            psnr_mv = self.psnr_fn(images_mv, images_mv_pred)
            psnr_mv_depth = self.psnr_fn(depths_mv, depths_mv_pred)
            if mode == 'val':
                log_dict.update({f"{mode}/PSNR_mv": psnr_mv})
                log_dict.update({f"{mode}/PSNR_mv_depth": psnr_mv_depth})

        # Log image when eval / snapshot
        if log_image_flag:
            # Log Singleview results
            # N = len(target_idx)
            images_mv_tolog =  postprocess_image(rearrange(images_mv, "(B n1 n2) C H W  -> B C (n1 H) (n2 W)", B = B, n1 = max(1, N // 4), n2 = N // max(1,  N // 4)), return_PIL = True)
            images_mv_pred_tolog = postprocess_image(rearrange(images_mv_pred, "(B n1 n2) C H W  -> B C (n1 H) (n2 W)", B = B, n1 = max(1, N // 4), n2 = N // max(1, N // 4)), return_PIL = True)


            depths_mv_gt_tolog = postprocess_image(rearrange(colorize_depth_maps(depths_mv, 0, 1,"Spectral_r"), "(B n1 n2) C H W  -> B C (n1 H) (n2 W)", B = B, n1 = max(1, N // 4), n2 = N // max(1,  N // 4)), 0, 1, return_PIL = True)
            depths_mv_pred_tolog = postprocess_image(rearrange(colorize_depth_maps(depths_mv_pred, 0, 1,"Spectral_r"), "(B n1 n2) C H W  -> B C (n1 H) (n2 W)", B = B, n1 = max(1, N // 4), n2 = N // max(1,  N // 4)), 0, 1, return_PIL = True)
            
            self.logger.log_image(f"snapshot_{mode}/multi_view_gt",images_mv_tolog, self.global_step, caption=texts)
            self.logger.log_image(f"snapshot_{mode}/multi_view",  images_mv_pred_tolog, self.global_step, caption=texts)
            # Log and visualize depth (single view)
            self.logger.log_image(f"snapshot_{mode}/multi_view_depth_gt", depths_mv_gt_tolog, self.global_step, caption=texts)
            self.logger.log_image(f"snapshot_{mode}/multi_view_depth_pred",depths_mv_pred_tolog, self.global_step, caption=texts)

            self.logger.log_text(key=f"snapshot_{mode}", 
                                  columns=['multi_view_text'], 
                                  data=[t for t in zip(texts)])            
            
            if self.use_gsdecoder and self.opt.losses.get('lambda_gs_image_mse',0) > 0:
                for k, v in gs_image_dict.items():
                    self.logger.log_image(k, v, self.global_step)
            
            # sample results from noise
            if True:
                inference_results = self.inference(
                    cameras=cameras,
                    text=texts,
                    ref_images = cond_images,
                    ref_idx = ref_idx,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    guidance_type='hybrid',
                    get_gs=False
                )

                if hasattr(self, 'gsdecoder'):
                    if input_latents.shape[1] == 8:
                        _idx = [0,2,5,7]
                    elif input_latents.shape[1] == 4:
                        _idx = [0,1,2,3]

                    with torch.no_grad():
                        _, _, gs_full_denoised_dict =  self.rendering_loss_fn(
                        latents=inference_results['latents_pred'][:,_idx],
                        cameras = cameras[:,_idx],
                        # gt images
                        cameras_gt=cameras[:],
                        # images denoise from unwt
                        log_image = True,
                        suffix='full_denoised',
                        rendering_size=256,
                        mode=mode)
                    
                    for k, v in gs_full_denoised_dict.items():
                        self.logger.log_image(k, v, self.global_step)

                images_mv_full_denoise = postprocess_image(rearrange(inference_results['images_pred'], "B (n1 n2) C H W  -> B C (n1 H) (n2 W)", B = B, n1 = max(1, N // 4), n2 = N // max(1,  N // 4)), return_PIL = True)
                depths_mv_full_denoise = postprocess_image(rearrange(colorize_depth_maps(inference_results['depths_pred'].flatten(0,1), 0, 1,"Spectral_r"), "(B n1 n2) C H W  -> B C (n1 H) (n2 W)", B = B, n1 = max(1, N // 4), n2 = N // max(1,  N // 4)), 0, 1, return_PIL = True)

                self.logger.log_image(f"snapshot_{mode}/multi_view_full_denoise", images_mv_full_denoise, self.global_step, caption=texts)
                self.logger.log_image(f"snapshot_{mode}/multi_view_full_denoise_depth",depths_mv_full_denoise, self.global_step, caption=texts)
    
        return loss_total, log_dict

    def inference_one_step(self, 
                        #    cameras, 
                           latents_noisy, 
                           text_embeddings, 
                           uncond_text_embeddings, 
                           pose_embeddings, 
                           uncond_pose_embeddings, 
                           t, 
                           guidance_type='hybrid', #['text', 'pose', 'joint', 'hybrid']
                           guidance_scale=1., 
                           pose_guidance_scale=1.,
                           use_3d_mode=False,
                           use_joint_guidance=True,
                           cfg_rescale=0.,
                           task_type = 'text_to_3d',
                           ref_latents = None,
                           ref_idx = None,
                           cameras = None,
                           gs_decoder = None,
                           ):
        """"""
        B, N, _, H ,W = latents_noisy.shape

        _t = t[..., None].repeat(1, N)
        _latents_noisy = latents_noisy.clone()

        if use_3d_mode:
            tt = _t.clone()
        elif guidance_type == 'text':
            # Naive text cfg as in Stable Diffusion, MVDream and Director3D
            tt = torch.cat([_t.clone()] * 2, 0)
            latents_noisy = torch.cat([latents_noisy.clone()] * 2, 0)
            text_embeddings = torch.cat([text_embeddings,
                                        uncond_text_embeddings], 0)
            pose_embeddings = torch.cat([pose_embeddings,
                                        pose_embeddings], 0)
        elif guidance_type == 'pose':
            tt = torch.cat([_t.clone()] * 2, 0)
            latents_noisy = torch.cat([latents_noisy.clone()] * 2, 0)
            text_embeddings = torch.cat([text_embeddings,
                                        text_embeddings], 0)
            pose_embeddings = torch.cat([pose_embeddings,
                                        uncond_pose_embeddings], 0)
        elif guidance_type == 'joint':
            # Pose-text Joint condition as in Zero 1-to-3 and ReconFusion
            tt = torch.cat([_t.clone()] * 2, 0)
            latents_noisy = torch.cat([latents_noisy.clone()] * 2, 0)
            text_embeddings = torch.cat([text_embeddings,
                                        uncond_text_embeddings], 0)
            pose_embeddings = torch.cat([pose_embeddings,
                                        uncond_pose_embeddings], 0)
        elif guidance_type == 'hybrid':
            # #TODO not work yet
            text_guidance_scale, pose_guidance_scale = 2 * guidance_scale / 3, guidance_scale / 3
            # text_guidance_scale, pose_guidance_scale = guidance_scale / 2, guidance_scale / 2
            tt = torch.cat([_t.clone()] * 3, 0)
            latents_noisy = torch.cat([latents_noisy.clone()] * 3, 0)
            
            text_embeddings = torch.cat([text_embeddings,
                                        uncond_text_embeddings, 
                                        text_embeddings], 0)
        
            pose_embeddings = torch.cat([
                                        pose_embeddings, 
                                        uncond_pose_embeddings, 
                                        uncond_pose_embeddings], 0)
        else:
            raise ValueError(f'Unsupport gudiance type {guidance_type}')
        


        # cat raymap
        if self.scheduler_type == 'edm' or \
            (self.scheduler_type  ==  'vprediction'):
            latents_noisy = self.scheduler.scale_model_input(latents_noisy, timestep=_t[0,0])

        if task_type == 'image_to_3d':
            latents_noisy[:B,ref_idx] = ref_latents.to(latents_noisy.dtype)
        
        latents_noisy = torch.cat([latents_noisy, pose_embeddings], dim = 2)
        latents_pred = self.model.denoise(
            latents_noisy,
            text_embeddings, 
            tt,)

        if use_3d_mode:
            _latents_pred = latents_pred
        elif guidance_type == 'hybrid':
            # Seprate CFG, might bug here
            cond_latents_pred, tuncond_latents_pred , puncond_latents_pred = latents_pred.chunk(3, dim=0)
            
            # _latents_pred = \
            # (cond_latents_pred - tuncond_latents_pred) * text_guidance_scale + \
            # (cond_latents_pred - puncond_latents_pred) * pose_guidance_scale +  (2 * tuncond_latents_pred + puncond_latents_pred) / 3

            # new
            _latents_pred = \
            tuncond_latents_pred + \
            (puncond_latents_pred - tuncond_latents_pred) * text_guidance_scale + \
            (cond_latents_pred - puncond_latents_pred) * pose_guidance_scale 
            
        else:
            cond_latents_pred, uncond_latents_pred = latents_pred.chunk(2, dim=0)
            _latents_pred = \
            (cond_latents_pred - uncond_latents_pred) * guidance_scale + \
            uncond_latents_pred

        # if use_3d_mode:
            # _latents_pred = cond_latents_pred

            # sigma = self.scheduler.sigmas[self.scheduler.step_index]
            # x0_pred = self.scheduler.precondition_outputs(_latents_noisy, _latents_pred, sigma)
            # x0_pred = torch.cat((x0_pred, pose_embeddings[:,:,:6]), dim = 2)


            # output_dict = self.scheduler.step(model_output=_latents_pred,
            #                     timestep=_t[0,0],
            #                     sample=_latents_noisy)
                    # clip or rescale x0
        if cfg_rescale > 0 and (not use_3d_mode):
            std_pos = cond_latents_pred.std([1,2,3,4], keepdim=True) 
            std_cfg = _latents_pred.std([1,2,3,4], keepdim=True) 
            # Apply guidance rescale with fused operations. factor = std_pos / std_cfg factor = rescale * factor + (1- rescale) return cfg * factor
            factor = std_pos / std_cfg
            factor = cfg_rescale * factor + (1- cfg_rescale) 
            _latents_pred = _latents_pred * factor 


        output_dict = self.scheduler.step(model_output=_latents_pred,
                            timestep=_t[0,0],
                            sample=_latents_noisy)
        
        
        if use_3d_mode:
            gamma = 0.4
            pred_original_sample = output_dict.pred_original_sample
            render_results = self.decoder_and_render(latents=pred_original_sample,
                                                     cameras = cameras, gs_decoder=gs_decoder)
            render_original_sample = self.render_to_latent(
                render_results['images_gs_render'],
                render_results['depths_gs_render'])

            sigma_from = self.scheduler.sigmas[self.scheduler.step_index - 1]
            sigma_to = self.scheduler.sigmas[self.scheduler.step_index]

            derivative = (_latents_noisy - render_original_sample) / sigma_from
            dt = sigma_to - sigma_from
            latents_less_noisy = _latents_noisy + derivative * dt
            # pred_original_sample = 
            # 2. Convert to an ODE derivative
            
        else:
            latents_less_noisy, pred_original_sample = output_dict.prev_sample, output_dict.pred_original_sample


        if task_type == 'image_to_3d':
            pred_original_sample[:B,ref_idx] = ref_latents.to(pred_original_sample.dtype)

        return latents_less_noisy, {'latents_pred': pred_original_sample}

    @torch.no_grad()
    def inference(self, 
                  cameras, 
                  text = None, 
                  ref_images = None,
                  ref_idx = None,
                  dense_cameras=None, 
                  inference_size = None, # H, W
                  gs_decoder_ext = None,
                  refiner=None, 
                  inference = None,
                  num_refine_steps=0,
                  num_inference_steps=25, 
                  guidance_scale=3.,
                  use_3d_mode_every_m_steps=-1, 
                  negative_text="", 
                  render_size=512,
                  guidance_type = 'hybrid',
                  cfg_rescale=0.7, # follow lin2024
                  get_gs = True, 
                  **kwargs):
        
        #TODO support multiple CFG (text / pose / view)
        B, N = cameras.shape[:2]
        if inference_size is None:
            H = W = self.latent_size
        elif isinstance(inference_size, int):
            H = W = inference_size // 8
        elif  isinstance(inference_size, tuple):
            H, W = inference_size[0] // 8, inference_size[1] // 8

        cameras = cameras.to(self.device)
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = self.scheduler.timesteps

        latents_noisy = torch.randn(B, N, self.latent_channel * 2, H, W, device=self.device) 
        
        # decoder for latent into 3d
        if gs_decoder_ext is not None:
            gs_decoder = gs_decoder_ext
        elif hasattr(self, 'gsdecoder'):
            gs_decoder = self.gsdecoder
        else:
            gs_decoder = None
            # raise ValueError("")

        # prepare cond & uncond ray embeddings
        rays_o, rays_d = sample_rays(cameras.flatten(0, 1), h=H, w=W, N=-1)# [B, N, H*W, 3]
        rays_embeddings = rearrange(embed_rays(rays_o, rays_d), "(B N) (H W) C -> B N C H W", B = B, H = H, W = W).contiguous().to(latents_noisy.device)

        uncond_rays_embeddings = torch.zeros_like(rays_embeddings)
        # if self.noise_type == 'view_cond':
        view_mask = torch.zeros_like(rays_embeddings[:,:,0:1])
        if self.task_type == 'image_to_3d':
            view_mask[:,ref_idx] = 1
        
        if self.noise_type == 'view_cond':
            rays_embeddings = torch.cat((rays_embeddings, view_mask), dim = 2)
            uncond_rays_embeddings =torch.cat((uncond_rays_embeddings, torch.zeros_like(rays_embeddings[:,:,0:1])), dim = 2)
        # else:
        #     uncond_rays_embeddings =torch.cat((uncond_rays_embeddings, torch.zeros_like(rays_embeddings[:,:,0:1])), dim = 2)

        # prepare cond & uncond text embeddings
        # TODO add image encoder
        if self.task_type == 'text_to_3d':
            text_embeddings = self.model.encode_text(text)
            uncond_text_embeddings =  self.model.encode_text(negative_text).repeat(B, 1, 1)
            ref_latents = None
        elif self.task_type == 'image_to_3d':
            text_embeddings = self.model.encode_image_clip(ref_images)
            uncond_text_embeddings =  self.model.encode_image_clip(torch.zeros_like(ref_images))

            # prepare ref latents
            ref_depths = self.get_depth_gt(ref_images, return_disp=True, normalize=True) 
            ref_depths_ = repeat(ref_depths * 2 - 1, "B N C H W -> B N (C c3) H W", c3 = 3)
            ref_latents  = torch.cat((self.model.encode_image(ref_images), self.model.encode_image(ref_depths_)), dim = 2)
            # latents_noisy[:,ref_idx] = ref_latents
        else:
            pass
        
        assert use_3d_mode_every_m_steps != 1, "use_3d_mode_every_m_steps can not be 1"
        if use_3d_mode_every_m_steps == -1:
              guidance_scale = guidance_scale
        else:
              guidance_scale = guidance_scale * use_3d_mode_every_m_steps / (use_3d_mode_every_m_steps - 1)
        
        if (self.scheduler_type =='edm') or \
            (self.scheduler_type  ==  'vprediction'):
            # scale the initial noise by the standard deviation required by the edm scheduler
            if self.task_type == 'text_to_3d' and self.ajust_sigma_on_N: # ajust sigma based on num of views
                self.scheduler.sigmas = self.scheduler.sigmas / math.sqrt(self.num_pred_views / N)
            
            # self.scheduler.sigmas = self.scheduler.sigmas / math.sqrt(8)    
            latents_noisy = latents_noisy * self.scheduler.init_noise_sigma

        for i, t in tqdm.tqdm(enumerate(timesteps), total=len(timesteps), desc='Denoising image sequence...'):
            if use_3d_mode_every_m_steps == -1:
                 use_3d_mode = False
            else:
                 use_3d_mode = not ((len(timesteps) - 1 - i) % use_3d_mode_every_m_steps)
            # use_3d_mode = False
            t = t[None].repeat(B)


            latents_noisy, result = self.inference_one_step(
            # cameras, 
            latents_noisy=latents_noisy, 
            text_embeddings=text_embeddings, 
            uncond_text_embeddings=uncond_text_embeddings, 
            pose_embeddings= rays_embeddings, 
            uncond_pose_embeddings= uncond_rays_embeddings, 
            t= t,
            guidance_scale=guidance_scale, 
            guidance_type=guidance_type,
            # pose_guidance_scale=pose_guidance_scale,
            use_3d_mode=use_3d_mode,
            cfg_rescale = cfg_rescale,
            task_type=self.task_type,
            ref_latents=ref_latents,
            ref_idx=ref_idx,
            cameras = cameras,
            gs_decoder = gs_decoder
            )

        # Decode 2d mv images
        latents_pred = result['latents_pred']
        images_pred = self.model.decode_latent(latents_pred[:,:,0:4],  mode='image')
        depths_pred = self.model.decode_latent(latents_pred[:,:,4:8],  mode='image').mean(dim = 2, keepdim=True) / 2 + 0.5
        result.update({
            'images_pred' : images_pred,
            'depths_pred' : depths_pred
            })


        if (gs_decoder is not None) and get_gs:
            Hl, Wl = latents_pred.shape[3], latents_pred.shape[4]
            rays_o, rays_d = sample_rays(cameras.flatten(0, 1), h=Hl, w=Wl, N=-1) # [B, N, H*W, 3]
            rays_embeddings = rearrange(embed_rays(rays_o, rays_d), "(B N) (H W) C -> B N C H W", B = B, H = Hl, W = Wl).to(latents_pred.device)

            latents_in = torch.cat((latents_pred, rays_embeddings), axis = 2)

            local_gaussian_params = gs_decoder.decode_latent(latents_in, mode = "gaussian")
            
            # if render in given camera views
            with torch.amp.autocast("cuda", enabled=False):
                gaussians = gs_decoder.converter(local_gaussian_params, cameras)
                images_nv_pred, depths_nv_pred, _, _, _ = gs_decoder.render( cameras, gaussians, h=render_size, w=render_size)
                result.update({
                    'cameras' : cameras,
                    'images_gs_render' : images_nv_pred,
                    'depths_gs_render' : depths_nv_pred,
                })

            result.update({"gaussians": gaussians})
        
        if refiner is not None:
            assert 'gaussians' in result
            assert dense_cameras is not None
            gaussians_sds = refiner.refine_gaussians(
                result['gaussians'], 
                text[0], 
                dense_cameras=dense_cameras)
            images_sds_pred, depth_sds_pred, _, _, _ = gs_decoder.render(cameras, gaussians_sds, h=render_size, w=render_size)
            result.update({
                "gaussians_sds": gaussians_sds, 
                'images_sds_gs_render' : images_sds_pred,
                'depths_sds_gs_render' : depth_sds_pred,
                })

        return result 
    
    def decoder_and_render(self, latents, cameras, gs_decoder, render_size = 256):
        result = {}
        B, N, C, Hl, Wl = latents.shape
        rays_o, rays_d = sample_rays(cameras.flatten(0, 1), h=Hl, w=Wl, N=-1) # [B, N, H*W, 3]
        rays_embeddings = rearrange(embed_rays(rays_o, rays_d), "(B N) (H W) C -> B N C H W", B = B, H = Hl, W = Wl).to(latents.device)

        latents_in = torch.cat((latents, rays_embeddings), axis = 2)

        local_gaussian_params = gs_decoder.decode_latent(latents_in, mode = "gaussian")
        
        # if render in given camera views
        with torch.amp.autocast("cuda", enabled=False):
            gaussians = gs_decoder.converter(local_gaussian_params, cameras)
            images_nv_pred, depths_nv_pred, _, _, _ = gs_decoder.render( cameras, gaussians, h=render_size, w=render_size)
        result.update({
            'cameras' : cameras,
            'images_gs_render' : images_nv_pred,
            'depths_gs_render' : depths_nv_pred,
        })

        return result
    
    def render_to_latent(self, images, depths):
        B, N, C, _, _ = images.shape
        depths_in =  depth_to_disp(depths.flatten(0,1), normalize=False)
        min_disps_ = depths_in.reshape(B, N, -1).min(dim=2)[0][:,:,None,None,None]
        max_disps_ = depths_in.reshape(B, N, -1).max(dim=2)[0][:,:,None,None,None]
        depths_in = 2 * (depths_in.repeat(1,3,1,1) - min_disps_) / (max_disps_- min_disps_) - 1
        images_in = images
        depth_latents = self.model.encode_image(depths_in)
        image_latents = self.model.encode_image(images_in)
        latents = torch.cat((image_latents, depth_latents), dim=2)
        return latents


    @staticmethod
    def get_raymap(cameras, H = 32, W = 32):
        B, N, _  = cameras.shape
        rays_o, rays_d = sample_rays(cameras.flatten(0, 1), h=H, w=W, N=-1)# [B, N, H*W, 3]
        raymap = rearrange(embed_rays(rays_o, rays_d), "(B N) (H W) C -> B N C H W", B = B, H = H, W = W).to(cameras.device)
        return raymap
        
    # #TODO GS rendering loss
    def rendering_loss_fn(self, 
                          latents, 
                          cameras, # camera of input views
                          images_gt = None, 
                          depths_gt = None, 
                          cameras_gt = None, 
                          log_image =True, 
                          rendering_size = 256, 
                          suffix = 'pred_latents',
                          view_mode = 'mv',
                          mode = 'train'):
        """Render GS and compute loss on input view"""
        gsdecoder = self.gsdecoder
        B, _, C, H, W = latents.shape
        loss_total = 0
        log_dict, img_dict = {}, {}
        if cameras_gt is None:
            # render at input view by default
            cameras_gt = cameras

        # depths_gt, images_gt = depths[:rendering_bs], images[:rendering_bs]
        raymap_gs_in = self.get_raymap(cameras, H = H, W = W)
        latents_gs_in = torch.cat((latents, raymap_gs_in), dim = 2)
        
        local_gaussian_params = gsdecoder.decode_latent(latents_gs_in, mode = "gaussian")
        gaussians = gsdecoder.converter(local_gaussian_params, cameras)
        # novel view synthesis
        with torch.amp.autocast("cuda", enabled=False):
            # images_nv = images[:, num_input_views:]
            # N = _images_nv.shape[1]
            images_gs, depths_gs, masks, reg_losses, states = gsdecoder.render(cameras_gt, \
            gaussians, 
            h=rendering_size,
            w=rendering_size)

        # Multiview loss (Novel view only)
    
        # loss_total += loss_mv_depth * self.opt.losses.lambda_mv_depth
        # loss_total += loss_mv_image_mse * self.opt.losses.lambda_mv_image_mse
        # loss_total += loss_mv_image_lpips * self.opt.losses.lambda_mv_image_lpips

        if images_gt is not None:
            lambda_gs_depth = self.opt.losses.get('lambda_gs_depth',1)
            lambda_gs_image_mse = self.opt.losses.get('lambda_gs_image_mse', 1)
            lambda_gs_image_lpips = self.opt.losses.get('lambda_gs_image_lpips',1)
            # 1. rgb mse loss
            loss_gs_image_mse = F.mse_loss(images_gs.flatten(0,1), images_gt.flatten(0,1))
            psnr_image = self.psnr_fn(images_gs.flatten(0,1), images_gt.flatten(0,1))
            # 2. midas depth loss
            mask = torch.ones_like(depths_gs)
            mask[depths_gs<=0.1] = 0 
            depths_gs_ = 1 / depths_gs.clip(0.1, 100)
            loss_gs_depth = self.disp_loss(prediction = depths_gs_.flatten(0,1)[:,0], target = depths_gt.flatten(0,1)[:,0], mask = mask.flatten(0,1)[:,0])
            # 3. lpips loss
            with torch.amp.autocast("cuda", enabled=False):    
                loss_gs_image_lpips = self.lpips_fn(
                    images_gs.flatten(0,1), 
                    images_gt.flatten(0,1)).mean()
            # 4. psnr
            psnr_image = self.psnr_fn(images_gt.flatten(0,1), images_gs.flatten(0,1))
            # Log mv loss 
            log_dict.update({f"{mode}/loss_gs_image_mse_{view_mode}" : loss_gs_image_mse})
            log_dict.update({f"{mode}/loss_gs_image_lpips_{view_mode}" : loss_gs_image_lpips})
            log_dict.update({f"{mode}/loss_gs_depth_{view_mode}" : loss_gs_depth})
            #log_dict.update({"val/loss_total" : loss_total})
            log_dict.update({f"{mode}/PSNR_gs_{view_mode}" : psnr_image})

            loss_total += loss_gs_depth * lambda_gs_depth
            loss_total += loss_gs_image_mse * lambda_gs_image_mse
            loss_total += loss_gs_image_lpips * lambda_gs_image_lpips

        if log_image:
            N = cameras_gt.shape[1]
            # Log Singleview results

            images_gs = rearrange(images_gs, "B (n1 n2) C H W  -> B C (n1 H) (n2 W)", B = B, n1 = max(1, N // 4), n2 = N // max(1, N // 4))
            # Log and visualize depth (multiview)
            # _depths_gs = rearrange(_depths_gs, "B N C H W -> (B N C) H W")
            depths_gs = rearrange(depths_gs,"B N C H W -> (B N C) H W")

            img_dict[f"snapshot_{mode}/gs_image_{suffix}"] = postprocess_image(images_gs, return_PIL = True)

            
            img_dict[f"snapshot_{mode}/gs_depth_{suffix}"] = postprocess_image(rearrange(colorize_depth_maps(depths_gs, None, None, "Spectral"), "(B n1 n2) C H W  -> B C (n1 H) (n2 W)", B = B, n1 = max(1, N // 4), n2 = N // max(1, N // 4)), 0, 1,return_PIL = True) # directly 
        

        return loss_total, log_dict, img_dict
    