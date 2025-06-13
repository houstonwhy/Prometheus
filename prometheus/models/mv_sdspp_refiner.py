import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from prometheus.modules.renderers.gaussians_renderer import GaussianRenderer
# from prometheus.utils.math import inverse_sigmoid
from prometheus.utils.camera import sample_from_dense_cameras
from prometheus.utils.gs_utils import GaussiansManeger
from prometheus.systems.depth_loss import depth_to_disp
from prometheus.utils import sample_rays, embed_rays

class MultiviewSDSPPRefiner(nn.Module):
    def __init__(self, 
            # sd_model_key='stabilityai/stable-diffusion-2-1-base',
            # use_mvsds=True,
            mvldm,
            scheduler,
            # local_files_only=True,
            num_views=1,
            total_iterations=500,
            # guidance_scale=100,
            min_step_percent=0.02, 
            max_step_percent=0.75,
            lr_scale=1,
            lr_scale_end=1,
            lrs={'xyz': 0.0001, 
                 'features': 0.01, 
                 'opacity': 0.05, 
                 'scales': 0.01, 
                 'rotations': 0.01}, 
            use_lods=True,
            lambda_latent_sds=1,
            lambda_image_sds=0.01,
            lambda_image_variation=0,
            lambda_mask_variation=0, 
            lambda_mask_saturation=0,
            use_depth_loss = False,
            use_random_background_color=True,
            grad_clip=10,
            img_size=256,
            num_densifications=5,
            text_templete='$text$',
            negative_text_templete='',
            guidance_scale=10,
            guidance_type = 'hybrid', # hybrid, text
            **kwargs
        ):
        super().__init__()

        # self.use_mvsds = use_mvsds
        self.guidance_type = guidance_type
        self.mvldm = mvldm
        self.use_depth_loss = use_depth_loss
        # if self.use_mvsds:
        self.guidance_scale= guidance_scale
        self.guidance_type = guidance_type

        self.tokenizer = mvldm.tokenizer
        self.text_encoder = mvldm.text_encoder.requires_grad_(False)
        self.vae = mvldm.vae.requires_grad_(False)
        self.unet = mvldm.unet.requires_grad_(False)
        
        # pipe = StableDiffusionPipeline.from_pretrained(
        #     sd_model_key, local_files_only=True
        # )
        # pipe.enable_xformers_memory_efficient_attention()

        # EDM scheduler related
        self.scheduler = scheduler
        sigma_data = 0.5
        self.c_skip = lambda sigma : sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        self.c_out = lambda sigma : sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
        self.c_in = lambda sigma : 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
        self.c_noise = lambda sigma : sigma.log() / 4
        self.weight = lambda sigma : (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
        # del pipe

        self.num_views = num_views
        self.total_iterations = total_iterations
        self.guidance_scale = guidance_scale
        self.lrs = {key: value * lr_scale for key, value in lrs.items()}
        self.lr_scale = lr_scale
        self.lr_scale_end = lr_scale_end


        self.device = 'cpu'

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.set_min_max_steps(min_step_percent, max_step_percent)

        self.renderer = GaussianRenderer()

        self.text_templete = text_templete
        self.negative_text_templete = negative_text_templete

        self.use_lods = use_lods

        self.lambda_latent_sds = lambda_latent_sds
        self.lambda_image_sds = lambda_image_sds
        self.lambda_image_variation = lambda_image_variation
        self.lambda_mask_variation = lambda_mask_variation
        self.lambda_mask_saturation = lambda_mask_saturation

        self.grad_clip = grad_clip
        self.img_size = img_size

        self.use_random_background_color = use_random_background_color

        self.opacity_threshold = 0.01
        self.densification_interval = self.total_iterations // (num_densifications + 1)

    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
    
    def to(self, device):
        self.device = device
        return super().to(device)

    @torch.no_grad()
    def encode_text(self, texts):
        text_embeddings = self.mvldm.encode_text(texts)
        return text_embeddings
    
    # @torch.amp.autocast("cuda", enabled=False)
    def encode_image(self, images):
        if len(images.shape) == 4:
            images = images[None]
        latents = self.mvldm.encode_image(images)
        # posterior = self.vae.encode(images).latent_dist
        # latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.squeeze(0)

    # @torch.amp.autocast("cuda", enabled=False)
    def decode_latent(self, latents, **kwargs):
        images = self.mvldm.decode_latent(latents, **kwargs)
        # latents = 1 / self.vae.config.scaling_factor * latents
        # images = self.vae.decode(latents).sample
        return images
    

    # TODO camera rays
    def get_rays_embeddings(self, cameras,latents, noise_type = 'vanilla'):
        N, _, h, w = latents.shape
        device = latents.device
        rays_o, rays_d = sample_rays(cameras, h=h, w=w, N=-1)# [B, N, H*W, 3]
        rays_embeddings = rearrange(embed_rays(rays_o, rays_d), "N (W H) C -> N C W H", N = N, H = h, W = w).to(device)
        if noise_type == 'view_cond':
            rays_embeddings = torch.cat((rays_embeddings, torch.zeros_like( rays_embeddings[:,:,0:1])), dim = 2)
        uncond_rays_embeddings = torch.zeros_like(rays_embeddings)

        return rays_embeddings, uncond_rays_embeddings
    
    def pre_diffusion(self, 
                      latents_noisy, 
                      t, 
                      rays_embeddings, 
                      uncond_rays_embeddings, 
                      text_embeddings, 
                      uncond_text_embeddings,
                      guidance_type = '' , 
                      guidance_scale = -1):

        if len(latents_noisy.shape) == 4: # N C H W
            t = t[None]
            latents_noisy = latents_noisy[None]
            text_embeddings = text_embeddings[None]
            uncond_text_embeddings = uncond_text_embeddings[None]
            rays_embeddings = rays_embeddings[None]
            uncond_rays_embeddings = uncond_rays_embeddings[None]

        if guidance_type == '':
            guidance_type = self.guidance_type
        if guidance_scale == -1:
            guidance_scale = self.guidance_scale
        
        if guidance_type == 'text':
            # Naive text cfg as in Stable Diffusion, MVDream and Director3D
            tt = torch.cat([t.clone()] * 2, 0)
            latents_noisy = torch.cat([latents_noisy.clone()] * 2, 0)
            text_embeddings = torch.cat([text_embeddings,
                                        uncond_text_embeddings], 0)
            rays_embeddings = torch.cat([rays_embeddings,
                                        rays_embeddings], 0)
        elif guidance_type == 'hybrid':
            # text_guidance_scale, pose_guidance_scale = 2 * guidance_scale / 3, guidance_scale / 3
            tt = torch.cat([t.clone()] * 3, 0)
            latents_noisy = torch.cat([latents_noisy.clone()] * 3, 0)
            
            text_embeddings = torch.cat([text_embeddings,
                                        uncond_text_embeddings, 
                                        text_embeddings], 0)
        
            rays_embeddings = torch.cat([
                                        rays_embeddings, 
                                        rays_embeddings, 
                                        uncond_rays_embeddings], 0)
        else:
            raise ValueError(f'Unsupport gudiance type {guidance_type}')
        
        latents_noisy = torch.cat((latents_noisy, rays_embeddings), dim = 2) # C
        tt = tt.to(latents_noisy.device)
        return latents_noisy, text_embeddings, tt

    def post_diffusion(self, latents_pred, latents_noisy, t, 
                       guidance_type = '' , guidance_scale = -1):
                       
        
        if guidance_scale == -1:
            guidance_scale = self.guidance_scale
        if guidance_type == '':
            guidance_type = self.guidance_type

        if guidance_type == 'text':
            cond_latents_pred, uncond_latents_pred = latents_pred.chunk(2, dim=0)
            _latents_pred = \
            (cond_latents_pred - uncond_latents_pred) * guidance_scale + \
            uncond_latents_pred
        elif guidance_type == 'hybrid':
            cond_latents_pred, tuncond_latents_pred, puncond_latents_pred = latents_pred.chunk(3, dim=0)
            pose_guidance_scale, text_guidance_scale  =  guidance_scale /  3,  2 * guidance_scale /  3
            _latents_pred = \
            (cond_latents_pred - tuncond_latents_pred) * text_guidance_scale + \
            (cond_latents_pred - puncond_latents_pred) * pose_guidance_scale + \
            cond_latents_pred
        else:
            raise ValueError(f'Unsupport guidance {guidance_type}')
        
        final_pred = self.scheduler.step(model_output=_latents_pred,
                        timestep=t[None],
                        sample=latents_noisy[None]).pred_original_sample[0]
        
        final_pred_nocfg = self.scheduler.step(model_output=cond_latents_pred,
                        timestep=t[None],
                        sample=latents_noisy[None]).pred_original_sample[0]
        
        # latents_less_noisy, pred_original_sample = output_dict.prev_sample, output_dict.pred_original_sample

        return final_pred, final_pred_nocfg
        
    def train_step(
        self,
        images,
        depths,
        cameras,
        t,
        text_embeddings,
        uncond_text_embeddings,
        learnable_text_embeddings,
        guidance_scale=10,
        guidance_type='hybrid'
        # use_rgbd = -1
    ):
        N, C, H, W = images.shape
        # TODO process depth: metric depth to disp [-1,1]
        # Encode multiview RGBD (Convert into inverse depth?)
        depths_in =  depth_to_disp(depths, normalize=False)
        min_disps_ = depths_in.reshape(N,-1).min(dim=1)[0][:,None,None,None]
        max_disps_ = depths_in.reshape(N,-1).max(dim=1)[0][:,None,None,None]
        depths_in = 2 * (depths_in.repeat(1,3,1,1) - min_disps_) / (max_disps_- min_disps_) - 1
        images_in = images
        depth_latents = self.encode_image(depths_in)
        image_latents = self.encode_image(images_in)
        latents = torch.cat((image_latents, depth_latents), dim=1)

        # Add noise on
        noise = torch.randn(latents.shape, device=latents.device) 
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        latents_noisy = latents_noisy * self.scheduler.init_noise_sigma # input precondition for edm 
        rays_embeddings , uncond_rays_embeddings = self.get_rays_embeddings(cameras, latents)

        # with torch.no_grad():
        N = latents.shape[0]
        t = t.repeat(N)
        
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t[0:1])
        latents_noisy = self.scheduler.scale_model_input(
            latents_noisy, 
            timestep=t[0:1])
        #TODO add noise in edm manner
        latents_noisy_in, text_embeddings_in, t_in = self.pre_diffusion(latents, t, rays_embeddings, uncond_rays_embeddings, text_embeddings, uncond_text_embeddings)
        #                                                                 guidance_scale=10,
        # guidance_type='hybrid')
    
        #TODO LOD edm version
        # if self.use_lods:
        #     with torch.enable_grad():
        #         noise_pred_learnable = self.unet(
        #             latents_noisy, 
        #             t, 
        #             encoder_hidden_states=learnable_text_embeddings
        #         ).sample
        loss_embedding = 0
        #     loss_embedding = F.mse_loss(noise_pred_learnable, noise, reduction="mean")
        # else:
        #     noise_pred_learnable = noise
        #     loss_embedding = 0

        with torch.no_grad():
            model_pred = self.mvldm.denoise(
            latents_noisy_in,
            text_embeddings_in, 
            t_in)
            # model_preds = self.mvldm.denoise(
            # latents_noisy,
            # text_embeddings, 
            # t_in,)     
            # noise_pred = self.unet(
            #     torch.cat([latents_noisy, latents_noisy], 0), 
            #     torch.cat([t, t], 0), 
            #     encoder_hidden_states=torch.cat([text_embeddings, uncond_text_embeddings], 0)
            # ).sample

        # c_skip, c_out, weight = self.c_skip, self.c_out, self.weight
        # sigmas = t
        latent_preds, latent_preds_nocfg = self.post_diffusion(
            model_pred, 
            latents_noisy, 
            t_in,        
            # guidance_scale=10,
            # guidance_type='hybrid'
            )
    
        # TODO add rescale as in MVDReam

        rgb_latents_pred, depth_latents_pred = \
            latent_preds[:,:4], latent_preds[:,4:]
        
        
        images_pred = self.decode_latent(rgb_latents_pred[None])[0]
        depths_pred = self.decode_latent(depth_latents_pred[None])[0]

        loss_latents_sds = ((rgb_latents_pred - image_latents)**2).sum()
        loss_images_sds = ((images_pred - images_in)**2.).sum()
        
        if self.use_depth_loss:
            loss_latents_sds += ((depth_latents_pred - depth_latents)**2.).sum()
            loss_images_sds += ((depths_pred - depths_in)**2.).sum()



        # TODO combine SDS + Recon as in MicroDreamer

        # noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
        # noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # w = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        # alpha = self.alphas_cumprod[t].view(-1, 1, 1, 1) ** 0.5
        # sigma = (1 - alpha) ** 0.5

        # latents_pred = (latents_noisy - sigma * (noise_pred - noise_pred_learnable + noise)) / alpha

        # loss_latent_sds = (F.mse_loss(latents, latents_pred, reduction="none").sum([1, 2, 3]) * w * alpha / sigma).sum() / B
        # loss_image_sds = (F.mse_loss(images, images_pred, reduction="none").sum([1, 2, 3]) * w * alpha / sigma).sum() / B

        
        return loss_latents_sds, loss_images_sds, loss_embedding

    @torch.amp.autocast("cuda", enabled=True)
    @torch.enable_grad()
    def refine_gaussians(self, gaussians, text, dense_cameras, num_views = -1):
        
        if num_views == -1:
            num_views = self.num_views
            
        gaussians_original = gaussians
        xyz, features, opacity, scales, rotations = gaussians

        mask = opacity[..., 0] >= self.opacity_threshold
        xyz_original = xyz[mask][None]
        features_original = features[mask][None]
        opacity_original = opacity[mask][None]
        scales_original = scales[mask][None]
        rotations_original = rotations[mask][None]

        text = self.text_templete.replace('$text$', text)

        text_embeddings = self.encode_text([text])
        uncond_text_embeddings =  self.encode_text([self.negative_text_templete.replace('$text$', text)])

        class LearnableTextEmbeddings(nn.Module):
            def __init__(self, uncond_text_embeddings):
                super().__init__()
                self.embeddings = nn.Parameter(torch.zeros_like(uncond_text_embeddings.float().detach().clone()))
                self.to(self.embeddings.device)

            def forward(self, cameras):
                B = cameras.shape[1]
                return self.embeddings.repeat(B, 1, 1)

        _learnable_text_embeddings = LearnableTextEmbeddings(uncond_text_embeddings)

        # text_embeddings = text_embeddings.repeat(num_views, 1, 1)
        # uncond_text_embeddings = uncond_text_embeddings.repeat(num_views, 1, 1)
        # text_embeddings = text_embeddings
        # uncond_text_embeddings = uncond_text_embeddings

        new_gaussians = GaussiansManeger(xyz_original, 
                                         features_original, 
                                         opacity_original, 
                                         scales_original, 
                                         rotations_original, self.lrs)

        optimizer_embeddings = torch.optim.Adam(_learnable_text_embeddings.parameters(), lr=self.lrs['embeddings'])

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(new_gaussians.optimizer, gamma=(self.lr_scale_end / self.lr_scale) ** (1 / self.total_iterations))
        self.scheduler.set_timesteps(1000)

        # self.scheduler.timesteps(self.scheduler.config.num_train_timesteps)

        for i in tqdm.trange(self.total_iterations, desc='Refining...'):

            if i % self.densification_interval == 0 and i != 0:
                new_gaussians.densify_and_prune()

            with torch.amp.autocast("cuda", enabled=False):   
                cameras = sample_from_dense_cameras(dense_cameras, 
                                                    torch.rand(1, num_views).to(self.device))

                learnable_text_embeddings = _learnable_text_embeddings(cameras)

                if self.lambda_mask_variation > 0 or self.lambda_image_variation > 0:
                    with torch.no_grad():
                        images_original, depths_original, masks_original, _, _ = self.renderer(cameras, gaussians_original, bg_color='random', h=self.img_size, w=self.img_size)

                gaussians = new_gaussians()
                images_pred, depths_pred, masks_pred, reg_losses, _ = self.renderer(cameras, gaussians, bg_color='random', h=self.img_size, w=self.img_size)
            
            
            t = torch.full((1,), int((i / self.total_iterations) ** (1/2) * (self.min_step - self.max_step) + self.max_step), dtype=torch.long, device=self.scheduler.timesteps.device)

            # if True:
            loss_latent_sds, loss_img_sds, loss_embedding = self.train_step(
                images = images_pred.squeeze(0), 
                depths = depths_pred.squeeze(0),
                cameras = cameras.squeeze(0),
                t = self.scheduler.timesteps[t], 
                text_embeddings = text_embeddings.squeeze(0),
                uncond_text_embeddings = uncond_text_embeddings.squeeze(0), learnable_text_embeddings = learnable_text_embeddings,
                # rays_embeddings = rays_embeddings,
                # uncond_rays_embeddingss = uncond_rays_embeddings, 
                )

            loss = loss_latent_sds * self.lambda_latent_sds + \
                  loss_img_sds * self.lambda_image_sds + loss_embedding

            if self.lambda_mask_variation > 0 or self.lambda_image_variation > 0:
                loss += self.lambda_mask_variation * F.mse_loss(masks_original, masks_pred, reduction='sum') / self.num_views
                loss += self.lambda_image_variation * F.mse_loss(images_original, images_pred, reduction='sum') / self.num_views

            if self.lambda_mask_saturation > 0:
                loss += self.lambda_mask_saturation * F.mse_loss(masks_pred, torch.ones_like(masks_pred), reduction='sum') / self.num_views

            # self.lambda_scale_regularization
            if True:
                scales = torch.sigmoid(new_gaussians._scales)
                big_points_ws = scales.max(dim=1).values > 0.1
                loss += 10 * scales[big_points_ws].sum()
                
            loss.backward()

            new_gaussians.optimizer.step()
            new_gaussians.optimizer.zero_grad()

            optimizer_embeddings.step()
            optimizer_embeddings.zero_grad()

            lr_scheduler.step()
            
            for radii, viewspace_points in zip(self.renderer.radii, self.renderer.viewspace_points):
                visibility_filter = radii > 0
                new_gaussians.is_visible[visibility_filter] = 1
                new_gaussians.max_radii2D[visibility_filter] = torch.max(new_gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                new_gaussians.add_densification_stats(viewspace_points, visibility_filter)
                # new_gaussians.xyz_gradient_accum[visibility_filter] += torch.norm(viewspace_points.grad[visibility_filter], dim=-1, keepdim=True)
        
        gaussians = new_gaussians()
        is_visible = new_gaussians.is_visible.bool()
        gaussians = [p[:, is_visible].detach() for p in gaussians]

        del new_gaussians
        return gaussians
    
# refiner_cfg = OmegaConf.load('configurations/refiner/mvsds.yaml')
# refiner = MultiviewSDSPPRefiner(**refiner_cfg['args'], mvldm=system.model, scheduler = system.scheduler, total_iterations=100).to(device)

# with torch.amp.autocast("cuda", enabled=True):
#     gaussians = result['gaussians']
#     gaussians = refiner.refine_gaussians(
#         gaussians, 
#         text, 
#         dense_cameras=dense_cameras[None])    
