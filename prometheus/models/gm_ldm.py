"""
Main GM-LDM Model
"""
#pylint: disable=import-error
#pylint: disable=not-callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from diffusers import StableDiffusionPipeline

from prometheus.modules.renderers.gaussians_renderer import GaussianRenderer, GaussianConverter
from prometheus.modules.unet_hacked import MultiViewUNetModel
from prometheus.modules.vae_hacked import AutoencoderKL
from prometheus.utils import sample_rays

class GMLDMModel(nn.Module):
    """XX"""
    def __init__(self, opt, mode='training'):
        """XX"""
        super().__init__()
        self.opt = opt
        self.mode=mode
        self.image_size = self.opt.network.image_size
        self.latent_size = self.opt.network.latent_size
        self.latent_channel = self.opt.network.latent_channel # 4

        pipe = StableDiffusionPipeline.from_pretrained(
            self.opt.network.sd_model_key,
            local_files_only=self.opt.network.local_files_only
        )

        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.requires_grad_(False)

        del pipe

        self.vae = AutoencoderKL(**self.opt.network.vae)
        self.vae_scale_factor = 0.18215

        self.latents_scale_fn = lambda x: x.sample() * self.vae_scale_factor
        self.latents_unscale_fn = lambda x: x * (1 / self.vae_scale_factor)

        self.vae.quant_conv.requires_grad_(False)
        self.vae.encoder.requires_grad_(False)

        self.vae.post_quant_conv.requires_grad_(True)
        self.vae.decoder.requires_grad_(True)

        self.unet = MultiViewUNetModel(**self.opt.network.unet).requires_grad_(True) # seems inherit from MVDream https://github.com/bytedance/MVDream/blob/main/mvdream/ldm/modules/diffusionmodules/openaimodel.py

        self.converter = GaussianConverter()
        self.renderer = GaussianRenderer()

        #self.num_input_views = self.opt.network.num_input_views

        self.initialize_weights()

    @torch.no_grad()
    def initialize_weights(self):
        """_summary_

        Init Dual3D_GS from sd_21_base ckpt
        """
        if self.opt.network.unet_pretrained_path:
            state_dict = torch.load(self.opt.network.unet_pretrained_path, map_location='cpu', weights_only = False)['state_dict']
            unet_state_dict = {}
            for key, value in state_dict.items():
                if 'model.diffusion_model.' in key:
                    unet_state_dict[key.replace('model.diffusion_model.', '')] = value

                # Repropose image decoder's output layer as GS Decoder
            self.unet.load_state_dict(unet_state_dict)
            self.unet.input_blocks[0][0].weight = nn.Parameter(F.pad(self.unet.input_blocks[0][0].weight, (0, 0, 0, 0, 0, self.opt.network.extra_latent_channel)))
            # [320, 4, 3, 3] -> [320, 388, 3, 3]?

            weight = F.pad(self.unet.out[-1].weight, (0, 0, 0, 0, 0, 0, 0, self.opt.network.extra_latent_channel))
            # [4, 320, 3, 3] -> [512, 320, 3, 3]
            # extra_latent_channel == 508 -> what's the meanning, channel num of F?
            weight[-self.opt.network.extra_latent_channel:] = torch.randn_like(weight[-self.opt.network.extra_latent_channel:]) * 0.01
            self.unet.out[-1].weight = nn.Parameter(weight.detach())
            self.unet.out[-1].bias = nn.Parameter(F.pad(self.unet.out[-1].bias, (0, self.opt.network.extra_latent_channel)))

            vae_state_dict = {}
            for key, value in state_dict.items():
                if 'first_stage_model.' in key:
                    vae_state_dict[key.replace('first_stage_model.', '')] = value

            self.vae.load_state_dict(vae_state_dict)
            # Repropose image decoder's output layer as GS Decoder
            self.vae.decoder.conv_in.weight = nn.Parameter(F.pad(self.vae.decoder.conv_in.weight, (0, 0, 0, 0, 0, self.opt.network.extra_latent_channel)))
            # [512, 4, 3, 3] -> [512, 512, 3, 3])
            weight = F.pad(self.vae.decoder.conv_out.weight, (0, 0, 0, 0, 0, 0, 0, sum(self.converter.gaussian_channels) - 3))
            self.vae.decoder.conv_out.weight = nn.Parameter(torch.zeros_like(weight.detach()))
            # [3, 128, 3, 3] -> [14, 512, 3, 3])
            self.vae.decoder.conv_out.bias = nn.Parameter(torch.zeros_like(F.pad(self.vae.decoder.conv_out.bias, (0, sum(self.converter.gaussian_channels) - 3))))

        else:
            raise ValueError('Can not load from SD21 ckpt')

        # EMANorm is the key to stabilize training if you wanna add pixel-wise rendering loss
        for i_level in reversed(range(self.vae.decoder.num_resolutions)):
            if i_level != 0:
                self.vae.decoder.up[i_level].upsample.conv = nn.Sequential(
                    self.vae.decoder.up[i_level].upsample.conv,
                    EMANorm(beta=0.995)
                )

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

    def encode_image(self, images):
        assert images.dim() == 5

        B, N = images.shape[:2]
        images = images.flatten(0, 1)
        latents = self.latents_scale_fn(self.vae.encode(images))
        latents = latents.unflatten(0, (B, N))
        return latents

    def decode_latent(self, latents, mode='gaussian'):
        assert latents.dim() == 5

        B, N = latents.shape[:2]
        latents = latents.flatten(0, 1)
        if mode=='gaussian':
            images = self.vae.decode(self.latents_unscale_fn(latents[:, :self.latent_channel]), extra_z=latents[:, self.latent_channel:])
        elif mode == 'image':
            images = self.vae.decode(self.latents_unscale_fn(latents[:, :self.latent_channel]), extra_z=latents[:, self.latent_channel:])
        
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
        return_3d=True,
        num_views=None,
    ):
        """XX"""
        B, N, _, _ ,_ = latents_noisy.shape

        if cameras is None:
            rays_embeddings = torch.zeros(B, N, 6 * (self.image_size//self.latent_size) ** 2, self.latent_size, self.latent_size).to(latents_noisy.device)
        else:
            rays_o, rays_d = sample_rays(cameras.flatten(0, 1), h=self.image_size, w=self.image_size, N=-1) # [B, N, H*W, 3]
            rays_embeddings = self.embed_rays(rays_o, rays_d).reshape(B, N, self.latent_size, self.image_size//self.latent_size, self.latent_size, self.image_size//self.latent_size, 6).permute(0, 1, 6, 3, 5, 2, 4).flatten(2, 4) # [B, N, 384, 32, 32]
            #TODO check if any bug here -> Try to directly sample low-res raymap? as in CAT3D and RayDiff?
        latents_noisy = torch.cat([latents_noisy, rays_embeddings], 2)

        assert text_embeddings.dim() == 3 and text_embeddings.shape[0] == B
        text_embeddings = text_embeddings.unsqueeze(1).repeat(1, N, 1, 1).flatten(0, 1)

        assert t.shape[0] == B and t.shape[1] == N
        t = t.flatten(0, 1)

        latents = self.unet(latents_noisy.flatten(0, 1), timesteps=t, context=text_embeddings, y=None, num_frames=N if num_views is None else num_views).unflatten(0, (B, N))

        latents2d_pred = latents[:, :, :self.latent_channel]

        if return_3d:
            local_gaussian_params = self.decode_latent(latents)
            gaussians = self.converter(local_gaussian_params, cameras)
            return latents2d_pred, gaussians

        return latents2d_pred

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
                self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.beta))
        input_gain = (self.magnitude_ema + 1e-5).rsqrt()
        x = x.mul(input_gain)
        return x


# class GaussianDrivenLDM(nn.Module):
#     def __init__(self, opt):
#         super().__init__()
#         self.opt = opt

#         self.image_size = self.opt.network.image_size
#         self.latent_size = self.opt.network.latent_size
#         self.latent_channel = self.opt.network.latent_channel

#         pipe = StableDiffusionPipeline.from_pretrained(
#             self.opt.network.sd_model_key, local_files_only=False
#         )
#         #self.opt.network.local_files_only
#         self.tokenizer = pipe.tokenizer
#         self.text_encoder = pipe.text_encoder.requires_grad_(False)

#         del pipe

#         self.vae = AutoencoderKL(**self.opt.network.vae)
#         self.vae_scale_factor = 0.18215

#         self.latents_scale_fn = lambda x: x.sample() * self.vae_scale_factor
#         self.latents_unscale_fn = lambda x: x * (1 / self.vae_scale_factor)

#         self.vae.quant_conv.requires_grad_(False)
#         self.vae.encoder.requires_grad_(False)

#         self.vae.post_quant_conv.requires_grad_(True)
#         self.vae.decoder.requires_grad_(True)

#         self.unet = MultiViewUNetModel(**self.opt.network.unet).requires_grad_(True)

#         self.converter = GaussianConverter()
#         self.renderer = GaussianRenderer()

#         self.num_input_views = self.opt.network.num_input_views

#         self.initialize_weights()

#     @torch.no_grad()
#     def initialize_weights(self):
#         self.unet.input_blocks[0][0].weight = nn.Parameter(F.pad(self.unet.input_blocks[0][0].weight, (0, 0, 0, 0, 0, 6 * (self.image_size // self.latent_size) ** 2)))

#         weight = F.pad(self.unet.out[-1].weight, (0, 0, 0, 0, 0, 0, 0, self.opt.network.extra_latent_channel))
#         weight[-self.opt.network.extra_latent_channel:] = torch.randn_like(weight[-self.opt.network.extra_latent_channel:]) * 0.01
#         self.unet.out[-1].weight = nn.Parameter(weight.detach())
#         self.unet.out[-1].bias = nn.Parameter(F.pad(self.unet.out[-1].bias, (0, self.opt.network.extra_latent_channel)))

#         self.vae.decoder.conv_in.weight = nn.Parameter(F.pad(self.vae.decoder.conv_in.weight, (0, 0, 0, 0, 0, self.opt.network.extra_latent_channel)))

#         weight = F.pad(self.vae.decoder.conv_out.weight, (0, 0, 0, 0, 0, 0, 0, sum(self.converter.gaussian_channels) - 3))
#         self.vae.decoder.conv_out.weight = nn.Parameter(torch.zeros_like(weight.detach()))
#         self.vae.decoder.conv_out.bias = nn.Parameter(torch.zeros_like(F.pad(self.vae.decoder.conv_out.bias, (0, sum(self.converter.gaussian_channels) - 3))))

#         for i_level in reversed(range(self.vae.decoder.num_resolutions)):
#             if i_level != 0:
#                 self.vae.decoder.up[i_level].upsample.conv = nn.Sequential(
#                     self.vae.decoder.up[i_level].upsample.conv,
#                     EMANorm(beta=0.995)
#                 )

#     @torch.no_grad()
#     def encode_text(self, texts):
#         inputs = self.tokenizer(
#             texts,
#             padding="max_length",
#             truncation_strategy='longest_first',
#             max_length=self.tokenizer.model_max_length,
#             return_tensors="pt",
#         )
#         text_embeddings = self.text_encoder(inputs.input_ids.to(next(self.text_encoder.parameters()).device))[0]
#         return text_embeddings

#     def encode_image(self, images):
#         assert images.dim() == 5

#         B, N = images.shape[:2]
#         images = images.flatten(0, 1)
#         latents = self.latents_scale_fn(self.vae.encode(images))
#         latents = latents.unflatten(0, (B, N))
#         return latents

#     def decode_latent(self, latents, mode='gaussian'):
#         assert latents.dim() == 5

#         B, N = latents.shape[:2]
#         latents = latents.flatten(0, 1)
#         images = self.vae.decode(self.latents_unscale_fn(latents[:, :self.latent_channel]), extra_z=latents[:, self.latent_channel:])
#         images = images.unflatten(0, (B, N))
#         return images

#     def embed_rays(self, rays_o, rays_d):
#         return torch.cat([rays_d, torch.cross(rays_o, rays_d, dim=-1)], -1)

#     def denoise(
#         self,
#         latents_noisy,
#         text_embeddings,
#         t,
#         cameras=None,
#         return_3d=True,
#         num_views=None,
#     ):
#         B, N, _, _ ,_ = latents_noisy.shape

#         if cameras is None:
#             rays_embeddings = torch.zeros(B, N, 6 * (self.image_size//self.latent_size) ** 2, self.latent_size, self.latent_size).to(latents_noisy.device)
#         else:
#             rays_o, rays_d = sample_rays(cameras.flatten(0, 1), h=self.image_size, w=self.image_size, N=-1)
#             rays_embeddings = self.embed_rays(rays_o, rays_d).reshape(B, N, self.latent_size, self.image_size//self.latent_size, self.latent_size, self.image_size//self.latent_size, 6).permute(0, 1, 6, 3, 5, 2, 4).flatten(2, 4)

#         latents_noisy = torch.cat([latents_noisy, rays_embeddings], 2)

#         assert text_embeddings.dim() == 3 and text_embeddings.shape[0] == B
#         text_embeddings = text_embeddings.unsqueeze(1).repeat(1, N, 1, 1).flatten(0, 1)

#         assert t.shape[0] == B and t.shape[1] == N
#         t = t.flatten(0, 1)

#         latents = self.unet(latents_noisy.flatten(0, 1), timesteps=t, context=text_embeddings, y=None, num_frames=N if num_views is None else num_views).unflatten(0, (B, N))

#         latents2d_pred = latents[:, :, :self.latent_channel]

#         if return_3d:
#             local_gaussian_params = self.decode_latent(latents)
#             gaussians = self.converter(local_gaussian_params, cameras)
#             return latents2d_pred, gaussians
#         else:
#             return latents2d_pred

#     def render(
        # self,
        # cameras,
        # gaussians,
        # **args):
        # results = self.renderer(cameras, gaussians, **args)

        # return results