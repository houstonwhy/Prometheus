"""
Main Multiview-LDM Model
"""
#pylint: disable=import-error
#pylint: disable=not-callable
import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from diffusers import StableDiffusionPipeline
from copy import deepcopy
# from prometheus.modules.renderers.gaussians_renderer import GaussianRenderer, GaussianConverter
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from prometheus.modules.unet_hacked import MultiViewUNetModel
from prometheus.modules.vae_hacked import AutoencoderKL
from prometheus.utils import sample_rays, embed_rays
from prometheus.utils.convert_diffuser_to_origin import convert_unet_state_dict, convert_vae_state_dict

additional_map = [
    ("time_embedding.linear_1.weight", "time_embed.0.weight"),
    ("time_embedding.linear_1.bias","time_embed.0.bias"), 
    ("time_embedding.linear_2.weight","time_embed.2.weight"), 
    ("time_embedding.linear_2.bias","time_embed.2.bias"), 
    ("conv_in.weight","input_blocks.0.0.weight"), 
    ("conv_in.bias","input_blocks.0.0.bias"), 
    ("conv_norm_out.weight","out.0.weight"), 
    ("conv_norm_out.bias","out.0.bias"), 
    ("conv_out.weight","out.2.weight"), 
    ("conv_out.bias","out.2.bias"),
]
class MVLDMModel(nn.Module):
    """XX"""
    def __init__(self, opt, mode='training'):
        """XX"""
        super().__init__()
        self.opt = opt
        self.mode=mode
        self.task_type = self.opt.get('task_type', 'text_to_3d')
        self.image_size = self.opt.image_size
        self.latent_size = self.opt.latent_size
        self.latent_channel = self.opt.latent_channel # ?
        self.extra_latent_channel = self.opt.extra_latent_channel

        pipe = StableDiffusionPipeline.from_pretrained(
            self.opt.sd_model_key,
            local_files_only=self.opt.local_files_only
        )
        self.unet_2d= pipe.unet
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.requires_grad_(False)
        self.vae_sd = pipe.vae.requires_grad_(False)
        self.vae_scale_factor = 0.18215
        self.latents_scale_fn = lambda x: x.sample() * self.vae_scale_factor
        self.latents_unscale_fn = lambda x: x * (1 / self.vae_scale_factor)
        del pipe
        if True:
            # new vae for load GS decoder
            self.vae = AutoencoderKL(**self.opt.vae)
            self.vae.quant_conv.requires_grad_(False)
            self.vae.encoder.requires_grad_(False)
            self.vae.post_quant_conv.requires_grad_(False)
            self.vae.decoder.requires_grad_(False)
        # new vae for load GS decoder
        # if self.opt.use_gsdecoder:
        #     self.gs_decoder = import_str(self.opt.algorithm.module)(opt, mode = mode)

        self.unet = MultiViewUNetModel(**self.opt.unet).requires_grad_(True) # seems inherit from MVDream https://github.com/bytedance/MVDream/blob/main/mvdream/ldm/modules/diffusionmodules/openaimodel.py
        
        # For image-2-3d model, we need additionally load CLIPImageEncoder
        if self.task_type == 'image_to_3d':
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                opt.image_encoder_path, 
                subfolder="feature_extractor", 
                # revision=args.revision
            )
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                opt.image_encoder_path, 
                subfolder="image_encoder", 
                # revision=args.revision
            ).requires_grad_(False)

        self.initialize_weights()

    @torch.no_grad()
    def initialize_weights(self):
        """_summary_

        Init MVLDM from sd_21_base ckpt
        """
        if self.mode == 'training':
            unet_state_dict = convert_unet_state_dict(self.unet_2d.state_dict())
            # Post process
            for o, t in additional_map:
                vv = unet_state_dict[o]
                del unet_state_dict[o]
                unet_state_dict[t] = vv
            # unet_state_dict.state_dict = self.unet.state_dict
            # Repropose image decoder's output layer as GS Decoder
            self.unet.load_state_dict(unet_state_dict, strict=True)
            # self.unet.input_blocks[0][0].weight = nn.Parameter(F.pad(self.unet.input_blocks[0][0].weight, (0, 0, 0, 0, 0, self.opt.extra_latent_channel)))
            
            weight_in = F.pad(self.unet.input_blocks[0][0].weight, (0, 0, 0, 0, 0, self.opt.extra_latent_channel)).detach()
            weight_out = F.pad(self.unet.out[-1].weight, (0, 0, 0, 0, 0, 0, 0, self.opt.extra_latent_channel)).detach()

            weight_in[:,-self.opt.extra_latent_channel:] = torch.randn_like(weight_in[:,-self.opt.extra_latent_channel:]) * 0.01
            weight_out[-self.opt.extra_latent_channel:] = torch.randn_like(weight_out[-self.opt.extra_latent_channel:]) * 0.01
            
            if self.opt.extra_latent_channel > 4:
                weight_in[:,4:8] = self.unet.input_blocks[0][0].weight.detach()
                weight_out[4:8] = self.unet.out[-1].weight.detach()
            
            self.unet.input_blocks[0][0].weight = nn.Parameter(weight_in.detach())
            self.unet.out[-1].weight = nn.Parameter(weight_out.detach())
            
            # self.unet.input_blocks[0][0].bias =  nn.Parameter(F.pad(self.unet.input_blocks[0][0].bias, (0, self.opt.extra_latent_channel)))
            self.unet.out[-1].bias = nn.Parameter(F.pad(self.unet.out[-1].bias, (0, self.opt.extra_latent_channel)))
            del self.unet_2d

            vae_state_dict = convert_vae_state_dict(self.vae_sd.state_dict())
            self.vae.load_state_dict(vae_state_dict, strict=True)
        
        else: # do not load from sd21
            if hasattr(self, 'unet'):
                # input 
                self.unet.input_blocks[0][0].weight = nn.Parameter(F.pad(self.unet.input_blocks[0][0].weight, (0, 0, 0, 0, 0, self.opt.extra_latent_channel)))
                weight = F.pad(self.unet.out[-1].weight, (0, 0, 0, 0, 0, 0, 0, self.opt.extra_latent_channel))
                weight[-self.opt.extra_latent_channel:] = torch.randn_like(weight[-self.opt.extra_latent_channel:]) * 0.01
                self.unet.out[-1].weight = nn.Parameter(weight.detach())
                self.unet.out[-1].bias = nn.Parameter(F.pad(self.unet.out[-1].bias, (0, self.opt.extra_latent_channel)))
    
    
    @torch.no_grad()
    def encode_image_clip(self,pixel_values):

        if len(pixel_values.shape) == 5:
            B, N, C, H, W = pixel_values.shape
            pixel_values = pixel_values.flatten(0,1)
            reshape_flag = 1
        else:
            reshape_flag = 0

        # pixel: [-1, 1]
        pixel_values = F.interpolate(pixel_values, (224, 224), mode="bicubic", align_corners=True, antialias=True)
        # We unnormalize it after resizing.
        pixel_values = (pixel_values + 1.0) / 2.0

        # Normalize the image with for CLIP input
        pixel_values = self.feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(device = self.image_encoder.device)
        image_embeddings = self.image_encoder(pixel_values).image_embeds
        if reshape_flag:
            image_embeddings = rearrange(image_embeddings, '(B N) C -> B N C', B = B, N= N)
        return image_embeddings
    

    @torch.no_grad()
    def encode_text(self, texts):
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation_strategy='longest_first',
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(inputs.input_ids.to(next(self.text_encoder.parameters()).device))[0]
        return text_embeddings

    # @torch.no_grad()
    def encode_image(self, images):
        assert images.dim() == 5

        B, N = images.shape[:2]
        images = images.flatten(0, 1)
        latents = self.latents_scale_fn(self.vae.encode(images))
        latents = latents.unflatten(0, (B, N))
        return latents
    
    def decode_latent(self, latents, mode='image', gsdecoder = None):
        assert latents.dim() == 5

        B, N, C, H, W = latents.shape
        latents = latents.flatten(0, 1)
        if mode=='gaussian':
            # raise ValueError("XX")
            images = self.vae_gs.decode(self.latents_unscale_fn(latents[:, :self.latent_channel]), extra_z=latents[:, self.latent_channel:])
        elif mode == 'image':
            assert C == 4
            images = self.vae_sd.decode(self.latents_unscale_fn(latents[:, :self.latent_channel]), return_dict=False)[0]
            # images = self.vae.decoder(self.latents_unscale_fn(latents[:, :self.latent_channel])) wrong, need vae.post_conv
        
        images = images.unflatten(0, (B, N))
        return images

    def embed_rays(self, rays_o, rays_d):
        return torch.cat([rays_d, torch.cross(rays_o, rays_d, dim=-1)], -1)

    def denoise(
        self,
        latents_noisy,
        text_embeddings,
        t,
        cameras=None,
        num_views=None,
        raymap_mode='lowres',
        return_3d=False,
    ):
        """XX"""
        B, N, C, H ,W = latents_noisy.shape

        if raymap_mode == 'none' or (cameras is None):
            pass
        # elif cameras is None:
        #     rays_embeddings = torch.zeros(B, N, 6, H, W).to(latents_noisy.device)
        #     latents_noisy = torch.cat([latents_noisy, rays_embeddings], 2)
        elif raymap_mode == 'highres': # high-res ray map in director3d
            rays_o, rays_d = sample_rays(cameras.flatten(0, 1), h=self.image_size, w=self.image_size, N=-1) # [B, N, H*W, 3]
            rays_embeddings = self.embed_rays(rays_o, rays_d).reshape(B, N, self.latent_size, self.image_size//self.latent_size, self.latent_size, self.image_size//self.latent_size, 6).permute(0, 1, 6, 3, 5, 2, 4).flatten(2, 4) # [B, N, 384, 32, 32]
            #TODO check if any bug here -> Try to directly sample low-res raymap? as in CAT3D and RayDiff?
            latents_noisy = torch.cat([latents_noisy, rays_embeddings], 2)
        elif raymap_mode == 'lowres':
            rays_o, rays_d = sample_rays(cameras.flatten(0, 1), h=H, w=W, N=-1)# [B, N, H*W, 3]
            rays_embeddings = rearrange(embed_rays(rays_o, rays_d), "(B N) (H W) C -> B N C H W", B = B, H = H, W = W).contiguous().to(latents_noisy.device)
            latents_noisy = torch.cat([latents_noisy, rays_embeddings], 2)
        else :
            raise ValueError('Unsupport raymap type')
        
        assert text_embeddings.dim() == 3 and text_embeddings.shape[0] == B
        text_embeddings = text_embeddings.unsqueeze(1).repeat(1, N, 1, 1).flatten(0, 1)

        # assert t.shape[0] == B and t.shape[1] == N
        t = t.flatten(0, 1)

        latents = self.unet(
            latents_noisy.flatten(0, 1), 
            timesteps=t, 
            context=text_embeddings, 
            y=None, 
            num_frames=N if num_views is None else num_views).unflatten(0, (B, N))

        latents = latents[:, :, :8] # rgbd latents

        # if return_3d:
        #     local_gaussian_params = self.decode_latent(latents)
        #     gaussians = self.converter(local_gaussian_params, cameras)
        #     return latents2d_pred, gaussians

        return latents

    def render(
        self,
        cameras,
        gaussians,
        **args):
        """XX"""
        results = self.renderer(cameras, gaussians, **args)

        return results




