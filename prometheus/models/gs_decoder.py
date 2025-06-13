"""
Main GM-LDM Model
"""
#pylint: disable=import-error
#pylint: disable=not-callable
import os
from copy import deepcopy
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
# from diffusers import StableDiffusionPipeline
from prometheus.modules.renderers.gaussians_renderer import GaussianRenderer, GaussianConverter
# from prometheus.modules.unet_hacked import MultiViewUNetModel
from prometheus.modules.ray_diff import DiT
from prometheus.modules.vae_hacked import AutoencoderKL
# from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import TemporalDecoder
# from prometheus.utils import sample_rays
class GSDecoderModel(nn.Module):
    """reproposing raw SD decoder as 3D GS Decoder"""
    def __init__(self, opt, use_ema_norm = False, mode='training'):
        """XX"""
        super().__init__()
        self.opt = opt
        self.mode=mode

        # self.opt = opt.network

        self.image_size = self.opt.image_size
        self.latent_size = self.opt.image_size // 8
        # self.latent_size = self.opt.latent_size
        self.latent_channel = self.opt.latent_channel # 4
        self.extra_latent_channel = self.opt.extra_latent_channel
        self.use_cross_view_dit = self.opt.use_cross_view_dit
        if self.use_cross_view_dit:
            self.cross_view_dit = DiT(**self.opt.cross_view_dit)
            #TODO multi layer down sample
            self.cross_view_dit.conv_in = Downsample(
                in_channels=self.latent_channel+self.extra_latent_channel ,
                out_channels=self.opt.cross_view_dit.in_channels, with_conv=True)
            self.cross_view_dit.conv_out = Upsample(
                in_channels=self.opt.cross_view_dit.out_channels,
                out_channels=self.latent_channel+self.extra_latent_channel, 
                with_conv=True)

            self.raydiff_pretrained_path = self.opt.raydiff_pretrained_path
        
        self.vae = AutoencoderKL(**self.opt.vae)

        #TODO Add temproal decoder from SVD
        self.vae_scale_factor = 0.18215
        self.latents_scale_fn = lambda x: x.sample() * self.vae_scale_factor

        self.latents_unscale_fn = lambda x: x * (1 / self.vae_scale_factor)

        self.vae.quant_conv.requires_grad_(False)
        self.vae.encoder.requires_grad_(False)

        self.vae.post_quant_conv.requires_grad_(True)
        self.vae.decoder.requires_grad_(True)

        self.converter = GaussianConverter(**opt.get('gs_converter', {}))
        self.renderer = GaussianRenderer(**opt.get('gs_renderer', {}))
        self.initialize_weights()

        # # EMANorm is the key to stabilize training if you wanna add pixel-wise rendering loss
        if self.opt.get('use_ema_norm', False) or use_ema_norm:
            for i_level in reversed(range(self.vae.decoder.num_resolutions)):
                if i_level != 0:
                    self.vae.decoder.up[i_level].upsample.conv = nn.Sequential(
                        self.vae.decoder.up[i_level].upsample.conv,
                        EMANorm(beta=0.995)
                    )
                    
    @torch.no_grad()
    def initialize_weights(self):
        """
        Init GS Decoder from sd21/svd/dit+sd21 VAE
       """
        
        if hasattr(self, 'raydiff_pretrained_path') and self.mode == 'training':
            try:
                if self.opt.debug:
                    with open(self.opt.raydiff_pretrained_path, 'rb') as f:
                        os.fsync(f.fileno())
                        state_dict = torch.load(f, map_location='cpu', weights_only=False)['state_dict']
                    # del ckpt_buffer
                else:
                    raydiff_state_dict = torch.load(self.opt.raydiff_pretrained_path, weights_only=False, map_location='cpu')['state_dict']
                raydiff_state_dict = {k.replace('ray_predictor.', '') : v for k, v in raydiff_state_dict.items() if 'ray_predictor.' in k}
                cur = self.cross_view_dit.state_dict()
                keys_to_delete = [key for key, param in raydiff_state_dict.items() 
                if key not in cur or param.size() != cur[key].size()]
                for key in keys_to_delete:
                    del raydiff_state_dict[key]
                #for k in raydiff_state_dict.keys():
                self.cross_view_dit.load_state_dict(raydiff_state_dict, strict=False)
            except:
                print(f'Load pretrained raydiff from {self.opt.raydiff_pretrained_path} failed')
            conv_padding_channels = self.opt.extra_latent_channel * 2 + 4
        elif self.use_cross_view_dit:
            conv_padding_channels = self.opt.extra_latent_channel * 2 + 4
        else:
            conv_padding_channels = self.opt.extra_latent_channel

        if self.opt.unet_pretrained_path and self.mode == 'training':
            state_dict = torch.load(self.opt.unet_pretrained_path, map_location='cpu', weights_only = False)['state_dict']
            # unet_state_dict = {}
            # for key, value in state_dict.items():
            #     if 'model.diffusion_model.' in key:
            #         unet_state_dict[key.replace('model.diffusion_model.', '')] = value
            vae_state_dict = {}
            for key, value in state_dict.items():
                if 'first_stage_model.' in key:
                    vae_state_dict[key.replace('first_stage_model.', '')] = value
            self.vae.load_state_dict(vae_state_dict)
        
        self.decoder_2d = deepcopy(self.vae.decoder)

        self.vae.decoder.conv_in.weight = nn.Parameter(F.pad(self.vae.decoder.conv_in.weight, (0, 0, 0, 0, 0, conv_padding_channels)))
        # [512, 4, 3, 3] -> [512, 512, 3, 3])
        self.vae.decoder.conv_out.weight = nn.Parameter(F.pad(self.vae.decoder.conv_out.weight, (0, 0, 0, 0, 0, 0, 0, sum(self.converter.gaussian_channels) - 3)))
        # [3, 128, 3, 3] -> [14, 512, 3, 3])
        self.vae.decoder.conv_out.bias = nn.Parameter(F.pad(self.vae.decoder.conv_out.bias, (0, sum(self.converter.gaussian_channels) - 3)))

    def encode_image(self, images):
        assert images.dim() == 5

        B, N = images.shape[:2]
        images = images.flatten(0, 1)
        latents = self.latents_scale_fn(self.vae.encode(images))
        latents = latents.unflatten(0, (B, N))
        return latents
    
    #@check_nan
    def decode_latent(self, latents, mode='gaussian'):
        assert latents.dim() == 5

        B, N = latents.shape[:2]
        latents = latents.flatten(0, 1)

        if mode == "gaussian":
            if not self.use_cross_view_dit:
                x_ = self.latents_unscale_fn(latents[:, :self.latent_channel])
                images = self.vae.decode(x_ , extra_z=latents[:, self.latent_channel:])
            else:
                x_ = self.cross_view_dit.conv_in(latents)
                x_ = rearrange(x_, '(B N) C H W -> B N C H W', B = B, N = N)
                x_ = self.cross_view_dit(x_, t=1000)
                x_ = rearrange(x_, 'B N C H W -> (B N) C H W')
                x_ = self.cross_view_dit.conv_out(x_)
                #x_ = self.latents_unscale_fn(latents[:, :self.latent_channel])
                x_  = torch.concatenate((latents, x_), dim = 1)
                images = self.vae.decode(x_[:,:4], extra_z=x_[:,4:], post_conv=True)
        elif mode == "2d":
            images = self.decoder_2d.decode(self.latents_unscale_fn(latents[:, :self.latent_channel]), extra_z=latents[:, self.latent_channel:])
        images = images.unflatten(0, (B, N))
        return images

    def embed_rays(self, rays_o, rays_d):
        return torch.cat([rays_d, torch.cross(rays_o, rays_d, dim=-1)], -1)


    def render(
        self,
        cameras,
        gaussians,
        **args):
        """XX"""
        results = self.renderer(cameras, gaussians, **args)

        return results

class EMANorm(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.register_buffer('magnitude_ema', torch.ones([]))
        self.beta = beta

    
    def forward(self, x):
        if self.training:
            magnitude_cur = x.detach().to(torch.float32).square().mean()
            if not magnitude_cur.isnan().any():
                self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema.to(torch.float32), self.beta))
        input_gain = (self.magnitude_ema + 1e-5).rsqrt()
        x = x.mul(input_gain)
        #print(input_gain)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

