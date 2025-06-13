"""
Trainer of GS-Decoder which take multi-view image as input -> output scene-level 3dGS
"""
#pyling:disable=import-error

import tqdm
import torch
import torch.nn.functional as F
from lightning import LightningModule
from .base_system import BaseSystem
from einops import rearrange, repeat
from prometheus.utils import sample_rays, embed_rays
from prometheus.utils.image_utils import postprocess_image, colorize_depth_maps

class GSDecoderSystem(BaseSystem):
    """GSDecoder system"""
    def __init__(self, opt, mode = 'training'):
        super().__init__(opt, mode)

        # set image_size as input & rendering
        self.input_size  = self.model.image_size
        self.target_size = self.opt.experiment.image_size

        if mode == 'training' and (not self.existing_checkpoint_path) :
            gsdecoder_ckpt_path = self.opt.training.get('gsdecoder_path', '')
            if gsdecoder_ckpt_path is not None:
                self.load_from_gsdecoder_ckpt(gsdecoder_ckpt_path)


    
    def load_from_gsdecoder_ckpt(self, gsdecoder_ckpt_path):
        """Load pretrained gsdecoder ckpt"""
        gsdecoder_ckpt = torch.load(gsdecoder_ckpt_path, weights_only=False, map_location='cpu')['state_dict']
        # Load GS-Decoder
        gsdecoder_state_dict = {}
        for key, value in gsdecoder_ckpt.items():
            if ('model_ema.' in key) and ('lpips_fn' not in key):
                gsdecoder_state_dict[key.replace('model_ema.', '')] = value
        try:
            self.model.load_state_dict(gsdecoder_state_dict,strict=True)
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
            self.model.load_state_dict(gsdecoder_state_dict,strict=False)
        del gsdecoder_ckpt 
        # del gsdecoder_state_dict
        print(f"Resume GS-Decoder form pretrained ckpt {gsdecoder_ckpt_path}. ")

    def forward_single_view(self, batch, mode = "val", batch_idx=0):
        """Single view prediction
        """
        loss_total, log_dict = 0, {}
        _images_sv, _texts2d  = batch["image"], batch["text"]
        _images_sv = _images_sv.flatten(0, 1).unsqueeze(1)
        B, N, C, H, W = _images_sv.shape
        input_size, target_size= self.input_size, self.target_size
        # B, N, C, H, W = _images.shape
        assert H == self.target_size
        _depths_sv = self.get_depth_gt(_images_sv, 
                                        return_disp=True, 
                                        normalize=True) # [B 1 1 H W]
        if H != input_size:
            images_sv = F.interpolate(_images_sv.flatten(0, 1), size=(input_size, input_size), align_corners=False, antialias=True, mode='bilinear').reshape(B, N, C, input_size, input_size)
            depths_sv = F.interpolate(_depths_sv.flatten(0, 1), size=(input_size, input_size), align_corners=False, antialias=True, mode='bilinear').reshape(B, N, 1, input_size, input_size)
        else:
            images_sv = _images_sv
            depths_sv = _depths_sv
        texts2d = []
        for text2d in _texts2d:
            texts2d += text2d
        # B, N, C, H, W = images_sv.shape
        #with torch.amp.autocast("cuda", enabled=False):
        #depths_sv, depths_sv_mask = depths_sv[:,:,0:1], depths_sv[:,:,1:2]
        depths_sv_in = repeat(depths_sv * 2 - 1, "B N C H W -> B N (C c3) H W", c3 = 3)
        input_latents2d = self.model.encode_image(images_sv)
        input_latents2d_depth = self.model.encode_image(depths_sv_in)

        Hl, Wl = input_latents2d.shape[3], input_latents2d.shape[4]
        #text_embeddings = self.model.encode_text(texts2d)
        if not 'cameras_mv' in batch.keys():
            cam_ = torch.tensor([1,0,0,0,
                                    0,1,0,0,
                                    0,0,1,0,
                                    207.3, 207.3, 
                                    128.0000, 128.0000, 
                                    256.0000, 256.0000]).to(images_sv)
            random_cameras = repeat(cam_, "C -> B N C", B = B, N = N)
        else:
            cameras =  batch['cameras_mv']
            random_cameras = cameras[:, 0:1].repeat(B//cameras.shape[0], 1, 1)
        random_rays_embeddings = torch.zeros(B, N, 6, Hl, Wl).to(images_sv.device)
        if self.decoder_mode == "RGB":
            latents_in = input_latents2d
        elif self.decoder_mode == "RGBD":
            latents_in = torch.cat((input_latents2d, input_latents2d_depth), axis = 2)
        elif self.decoder_mode == "RGBPose":
            latents_in = torch.cat((input_latents2d, random_rays_embeddings), axis = 2)
        elif self.decoder_mode == "RGBDPose":
            latents_in = torch.cat((input_latents2d, input_latents2d_depth,random_rays_embeddings), axis = 2)
    
        local_gaussian_params = self.model.decode_latent(latents_in, mode = "gaussian")
        gaussians2d = self.model.converter(local_gaussian_params, random_cameras)
        with torch.amp.autocast("cuda", enabled=False):
            images_sv_pred, depths_sv_pred, _masks, _, _ = self.model.render(random_cameras, gaussians2d, h=target_size, w=target_size,bg_color = self.render_bg_color)
        _images_sv = rearrange(_images_sv,"B N C H W -> (B N) C H W")
        images_sv_pred = rearrange(images_sv_pred,"B N C H W -> (B N) C H W")
            
        loss_sv_image_mse = F.mse_loss(_images_sv, images_sv_pred)
        loss_sv_image_lpips = self.lpips_fn(_images_sv, images_sv_pred).mean()
        loss_total += loss_sv_image_mse * self.opt.losses.lambda_sv_image_mse
        loss_total += loss_sv_image_lpips * self.opt.losses.lambda_sv_image_lpips
        PSNR_sv_image = self.psnr_fn(_images_sv, images_sv_pred)

        
        mask = torch.ones_like(depths_sv_pred)
        mask[depths_sv_pred<=0.1] = 0 
        depths_sv_pred_ = 1 / depths_sv_pred.clip(0.1, 100)
        loss_sv_depth = self.disp_loss(
            prediction = depths_sv_pred_.flatten(0,1)[:,0], 
            target = _depths_sv.flatten(0,1)[:,0], 
            mask = mask.flatten(0,1)[:,0])
        #loss_sv_depth, = depth_loss(depths_sv.flatten(0,1)[0], depths_sv_pred.flatten(0,1)[0])  # B H W

        loss_total += loss_sv_depth * self.opt.losses.lambda_sv_depth
        depths_sv = rearrange(depths_sv, "B N C H W -> (B N C) H W")
        depths_sv_pred = rearrange(depths_sv_pred,"B N C H W -> (B N C) H W")

        # Log mv loss 
        log_dict.update({f"{mode}/PSNR_sv" : PSNR_sv_image})
        log_dict.update({f"{mode}/loss_sv_image_mse" : loss_sv_image_mse})
        log_dict.update({f"{mode}/loss_sv_image_lpips":loss_sv_image_lpips})
        log_dict.update({f"{mode}/loss_sv_depth": loss_sv_depth})

        if self.opt.losses.get('lambda_entropy', 0.) > 0:
            loss_sv_entropy = (1 - _masks).mean()
            # opacity = gaussians2d[2]
            # loss_sv_entropy = (-opacity * torch.log(opacity))[opacity > 1e-4].mean() 
            log_dict.update({f"{mode}/loss_sv_entropy" : loss_sv_entropy})
            loss_total += self.opt.losses.lambda_entropy * loss_sv_entropy  

        # Log image when eval / snapshot
        if (mode == 'val' and batch_idx == 0) or \
        (mode == 'train' and self.global_step % self.log_every_n_step == 0):
            self.logger.log_image(f"snapshot_{mode}/single_view_gt", postprocess_image(_images_sv, return_PIL = True), self.global_step)
            self.logger.log_image(f"snapshot_{mode}/single_view",  postprocess_image(images_sv_pred, return_PIL = True), self.global_step)
            # Log and visualize depth (single view)
            self.logger.log_image(f"snapshot_{mode}/single_view_depth_gt",  postprocess_image(colorize_depth_maps(_depths_sv, 0, 1, "Spectral_r"), 0, 1, return_PIL = True), self.global_step)
            self.logger.log_image(f"snapshot_{mode}/single_view_depth_pred",  postprocess_image(colorize_depth_maps(depths_sv_pred, None, None, "Spectral"), 0, 1, return_PIL = True), self.global_step)# directly visualiza metric depth
            self.logger.log_image(f"snapshot_{mode}/single_view_depth_pred_disp",  postprocess_image(colorize_depth_maps(depths_sv_pred_, 0, 5, "Spectral_r"), 0, 1, return_PIL = True), self.global_step)# visualiza disparity

        return loss_total, log_dict
    
    def forward_multi_view(self, batch, mode = "val", batch_idx=0):
        loss_total = 0.
        log_dict = {}
        _images, cameras, texts  = batch["images_mv"], batch["cameras_mv"], batch["text_mv"]
        input_size, target_size, num_input_views, num_novel_views = self.input_size, self.target_size, self.num_input_views, self.num_novel_views
        B, N, C, H, W = _images.shape
        assert H == target_size
        _depths = self.get_depth_gt(_images, 
                                    return_disp=True, 
                                    normalize=True) # [B N 1 H W]
        _depths_iv, _depths_nv = _depths[:, :num_input_views], _depths[:, num_input_views:] # full-res depthgt
        _images_iv, _images_nv = _images[:, :num_input_views], _images[:, num_input_views:] # full-res image
        
        if H != input_size: # model input
            images = F.interpolate(_images.flatten(0,1), size=(input_size, input_size), align_corners=False, antialias=True, mode='bilinear').reshape(B, N, C, input_size, input_size)
            depths = F.interpolate(_depths.flatten(0,1), size=(input_size, input_size), align_corners=False, antialias=True, mode='bilinear').reshape(B, N, 1, input_size, input_size)
        else:
            images = _images
            depths = _depths
        
        # input & pred
        depths_in = repeat(depths[:,:num_input_views] * 2 - 1, "B N C H W -> B N (C c3) H W", c3 = 3)
        input_latents = self.model.encode_image(images[:,:num_input_views])
        input_latents_depth = self.model.encode_image(depths_in)
        Hl, Wl = input_latents.shape[3], input_latents.shape[4]
        rays_o, rays_d = sample_rays(cameras.flatten(0, 1), h=Hl, w=Wl, N=-1) # [B, N, H*W, 3]
        rays_embeddings = rearrange(embed_rays(rays_o, rays_d), "(B N) (H W) C -> B N C H W", B = B, H = Hl, W = Wl)
        
        if self.decoder_mode == "RGB":
            latents_in = input_latents
        elif self.decoder_mode == "RGBD":
            latents_in = torch.cat((input_latents, input_latents_depth), axis = 2)
        elif self.decoder_mode == "RGBPose":
            latents_in = torch.cat((input_latents, rays_embeddings[:,:num_input_views]), axis = 2)
        elif self.decoder_mode == "RGBDPose":
            latents_in = torch.cat((input_latents, input_latents_depth, rays_embeddings[:,:num_input_views]), axis = 2)

        local_gaussian_params = self.model.decode_latent(latents_in, mode = "gaussian")
        gaussians = self.model.converter(local_gaussian_params, cameras[:, :num_input_views])

        # novel view synthesis
        with torch.amp.autocast("cuda", enabled=False):
            # images_nv = images[:, num_input_views:]
            N = _images_nv.shape[1]
            images_nv_pred, depths_nv_pred, _masks, reg_losses, states = self.model.render(cameras[:, num_input_views:], \
            gaussians, h=target_size, w=target_size, bg_color = self.render_bg_color)

            _images_iv = rearrange(_images_iv,"B N C H W -> (B N) C H W")
            _images_nv = rearrange(_images_nv,"B N C H W -> (B N) C H W")
            images_nv_pred = rearrange(images_nv_pred,"B N C H W -> (B N) C H W")
            # Multiview loss (Novel view only)
            
            loss_mv_image_lpips = self.lpips_fn(images_nv_pred, _images_nv).mean()

            mask = torch.ones_like(depths_nv_pred)
            mask[depths_nv_pred<=0.1] = 0 
            depths_nv_pred_ = 1 / depths_nv_pred.clip(0.1, 100)
            loss_mv_depth = self.disp_loss(prediction = depths_nv_pred_.flatten(0,1)[:,0], target = _depths_nv.flatten(0,1)[:,0], mask = mask.flatten(0,1)[:,0])
    
        loss_mv_image_mse = F.mse_loss(images_nv_pred, _images_nv)
        loss_total += loss_mv_depth * self.opt.losses.lambda_mv_depth
        loss_total += loss_mv_image_mse * self.opt.losses.lambda_mv_image_mse
        loss_total += loss_mv_image_lpips * self.opt.losses.lambda_mv_image_lpips

        psnr_image = self.psnr_fn(images_nv_pred, _images_nv)
        # Log mv loss 
        log_dict.update({f"{mode}/loss_mv_image_mse" : loss_mv_image_mse})
        log_dict.update({f"{mode}/loss_mv_image_lpips" : loss_mv_image_lpips})
        log_dict.update({f"{mode}/loss_mv_depth" : loss_mv_depth})
        #log_dict.update({"val/loss_total" : loss_total})
        log_dict.update({f"{mode}/PSNR_mv" : psnr_image})

        if self.opt.losses.get('lambda_entropy', 0.) > 0:
            loss_mv_entropy = (1 - _masks).mean()
            # opacity = gaussians[2]
            # loss_mv_entropy = (-opacity * torch.log(opacity))[opacity > 1e-4].mean() 
            log_dict.update({f"{mode}/loss_mv_entropy" : loss_mv_entropy})
            loss_total += self.opt.losses.lambda_entropy * loss_mv_entropy  

        # Log image when eval / snapshot
        if (mode == "val" and batch_idx == 0) or \
        (mode == "train" and self.global_step % self.log_every_n_step == 0):
            # Log results
            _images_nv = rearrange(_images_nv, "(B n1 n2) C H W  -> B C (n1 H) (n2 W)", B = B, n1 = max(1, N // 2), n2 = N // max(1, N // 2))
            images_nv_pred = rearrange(images_nv_pred, "(B n1 n2) C H W  -> B C (n1 H) (n2 W)", B = B, n1 = max(1, N // 2), n2 = N // max(1, N // 2))
            # Log and visualize depth (multiview)
            _depths_nv_gt = rearrange(_depths_nv, "B N C H W -> (B N C) H W")
            depths_nv_pred = rearrange(depths_nv_pred,"B N C H W -> (B N C) H W")
            depths_nv_pred_ = rearrange(depths_nv_pred_,"B N C H W -> (B N C) H W")

            self.logger.log_image(f"snapshot_{mode}/novel_view_gt", postprocess_image(_images_nv, return_PIL = True),self.global_step)
            self.logger.log_image(f"snapshot_{mode}/novel_view", postprocess_image(images_nv_pred, return_PIL = True),self.global_step)
            self.logger.log_image(f"snapshot_{mode}/novel_view_depth_gt", postprocess_image(rearrange(colorize_depth_maps(_depths_nv_gt, 0, 1,"Spectral_r"), "(B n1 n2) C H W  -> B C (n1 H) (n2 W)", B = B, n1 = max(1, N // 2), n2 = N // max(1, N // 2)), 0, 1,return_PIL = True),self.global_step)
            self.logger.log_image(f"snapshot_{mode}/novel_view_depth_pred", postprocess_image(rearrange(colorize_depth_maps(depths_nv_pred, None, None,"Spectral"), "(B n1 n2) C H W  -> B C (n1 H) (n2 W)", B = B, n1 = max(1, N // 2), n2 = N // max(1, N // 2)), 0, 1,return_PIL = True),self.global_step) # directly visualiza metric depth
            self.logger.log_image(f"snapshot_{mode}/novel_view_depth_pred_disp", postprocess_image(rearrange(colorize_depth_maps(depths_nv_pred_, 0, 5, "Spectral_r"), "(B n1 n2) C H W  -> B C (n1 H) (n2 W)", B = B, n1 = max(1, N // 2), n2 = N // max(1, N // 2)), 0, 1,return_PIL = True),self.global_step)# disp
            
        return loss_total, log_dict

    def training_step(self, batch):
        self.lpips_fn.eval().requires_grad_(False)
        self.disp_fn.eval().requires_grad_(False)
        self.model.decoder_2d.eval().requires_grad_(False)
        self.model.vae.encoder.eval().requires_grad_(False)
        self.model.vae.decoder.train()
        self.update_average(self.model_ema, self.model)

        log_dict = {'step':self.global_step}
        loss_total = 0
        # Multi-view part
        if 'images_mv' in  batch.keys():
            loss_, log_dict_= self.forward_multi_view(batch=batch, mode='train')
            loss_total += loss_
            log_dict.update(log_dict_)
        # Single-view part
        if 'image' in  batch.keys() and self.single_view_num > 0:
            loss_, log_dict_ = self.forward_single_view(batch=batch, mode='train')
            loss_total += loss_
            log_dict.update(log_dict_)
       
        log_dict.update({'train/loss_total':loss_total})
        self.log_dict(log_dict, rank_zero_only=True)
        return loss_total

  
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # self.model 
        loss_total = 0.
        log_dict = {}
        #Multiview part
        if 'images_mv' in  batch.keys():
            loss_, log_dict_ = self.forward_multi_view(batch=batch, mode = 'val', batch_idx=batch_idx)
            loss_total += loss_
            log_dict.update(log_dict_)
        #Singleview part:
        #rearrange(images_sv, "B N C H W")
        if 'image' in  batch.keys():
            loss_, log_dict_ = self.forward_single_view(batch=batch, mode = 'val', batch_idx=batch_idx)
            loss_total += loss_
            log_dict.update(log_dict_)

        log_dict.update({'val/loss_total':loss_total})
        self.logger.log_metrics(log_dict, step=self.global_step)
        return
    
    @torch.amp.autocast("cuda", enabled=True)
    @torch.no_grad()
    def inference(self, cameras, images, num_input_views = -1, render_size =-1, mode = 'multi_view', bg_color = (1,1,1)):
        """
        GS Decoder inference -> predict 3dgs for single/multiview input image
        """
        images, cameras = images.to(self.device), cameras.to(self.device)
        if len(images.shape) == 5:
            B, N, C, H, W = images.shape
        elif len(images.shape) == 4 and mode =='multi_view':
            N, C, H, W = images.shape
            B = 1
            images, cameras = images[None], cameras[None]
        elif len(images.shape) == 4 and mode =='single_view':
            B, C, H, W = images.shape
            N = 1
            images, cameras = images[:,None], cameras[:,None]
        else:
            raise ValueError('Unsupport input format')
        
        if render_size == -1:
            render_size = self.target_size
        if num_input_views == -1:
            num_input_views = N
        cameras_iv = cameras[:,:num_input_views]
        cameras_nv = cameras[:,num_input_views:]

        depths_in = self.get_depth_gt(images[:,:num_input_views], 
                                    return_disp=True, 
                                    normalize=True) # [B N 1 H W]
        # depths_iv, depths_nv = depths[:, :self.num_input_views], depths[:, self.num_input_views:]
        depths_in = repeat(depths_in * 2 - 1, "B N C H W -> B N (C c3) H W", c3 = 3)
        input_latents = self.model.encode_image(images[:,:num_input_views])
        input_latents_depth = self.model.encode_image(depths_in)
        Hl, Wl = input_latents.shape[3], input_latents.shape[4]
        rays_o, rays_d = sample_rays(cameras.flatten(0, 1), h=Hl, w=Wl, N=-1) # [B, N, H*W, 3]
        rays_embeddings = rearrange(embed_rays(rays_o, rays_d), "(B N) (H W) C -> B N C H W", B = B, H = Hl, W = Wl)
        
        if self.decoder_mode == "RGB":
            latents_in = input_latents
        elif self.decoder_mode == "RGBD":
            latents_in = torch.cat((input_latents, input_latents_depth), axis = 2)
        elif self.decoder_mode == "RGBPose":
            latents_in = torch.cat((input_latents, rays_embeddings[:,:num_input_views]), axis = 2)
        elif self.decoder_mode == "RGBDPose":
            latents_in = torch.cat((input_latents, input_latents_depth, rays_embeddings[:,:num_input_views]), axis = 2)

        local_gaussian_params = self.model.decode_latent(latents_in, mode = "gaussian")
        gaussians = self.model.converter(local_gaussian_params, cameras_iv)
        result = {
            'gaussians':gaussians,
            "images_in":images[:,:num_input_views],
            "rays_embeddings":rays_embeddings[:,:num_input_views],
            "depths_in":depths_in,
            "latents_in":latents_in,
            "latents_crossview":latents_in
            }
        
        # if novel pose was given -> render novel view
        if len(cameras_nv[0]) > 0:
            with torch.amp.autocast("cuda", enabled=False):
                images_nv_pred, depths_nv_pred, _, _, _ = self.model.render( cameras_nv, gaussians, h=render_size, w=render_size, bg_color = bg_color)
                result.update({
                    'cameras_nv' : cameras_nv,
                    'images_nv_pred' : images_nv_pred,
                    'depths_nv_pred' : depths_nv_pred,
                })
        return result 




