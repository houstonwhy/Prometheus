# Borrow from https://github.com/jasonyzhang/RayDiffusion/blob/main/ray_diffusion/model/diffuser.py
import io
import ipdb  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .dit import DiT

# from ray_diffusion.model.feature_extractors import SpatialDino
# from ray_diffusion.model.scheduler import NoiseScheduler


class RayDiffuser(nn.Module):
    def __init__(
        self,
        model_type="dit",
        depth=8,
        width=16,
        hidden_size=1152,
        P=1,
        max_num_images=1,
        noise_scheduler=None,
        freeze_encoder=True,
        feature_extractor="dino",
        append_ndc=True,
        use_unconditional=False,
    ):
        super().__init__()
        if noise_scheduler is None:
            self.noise_scheduler = NoiseScheduler()
        else:
            self.noise_scheduler = noise_scheduler

        self.ray_dim = 6

        self.append_ndc = append_ndc
        self.width = width

        self.max_num_images = max_num_images
        self.model_type = model_type
        self.use_unconditional = use_unconditional

        if feature_extractor == "dino":
            self.feature_extractor = SpatialDino(
                freeze_weights=freeze_encoder, num_patches_x=width, num_patches_y=width
            )
            self.feature_dim = self.feature_extractor.feature_dim
        else:
            raise Exception(f"Unknown feature extractor {feature_extractor}")

        if self.use_unconditional:
            self.register_parameter(
                "null_token", nn.Parameter(torch.randn(self.feature_dim, 1, 1))
            )

        self.input_dim = self.ray_dim + self.feature_dim
        if self.append_ndc:
            self.input_dim += 2

        if model_type == "dit":
            self.ray_predictor = DiT(
                in_channels=self.input_dim,
                out_channels=self.ray_dim,
                width=width,
                depth=depth,
                hidden_size=hidden_size,
                max_num_images=max_num_images,
                P=P,
            )
        else:
            raise Exception(f"Unknown model type {model_type}")

    def forward_noise(self, x, t, epsilon=None, mask=None):
        """
        Applies forward diffusion (adds noise) to the input.

        If a mask is provided, the noise is only applied to the masked inputs.
        """
        t = t.reshape(-1, 1, 1, 1, 1)
        if epsilon is None:
            epsilon = torch.randn_like(x)
        else:
            epsilon = epsilon.reshape(x.shape)
        alpha_bar = self.noise_scheduler.alphas_cumprod[t]
        x_noise = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * epsilon
        if mask is not None:
            x_noise = x_noise * mask + x * (1 - mask)
        return x_noise, epsilon

    def forward(
        self,
        features=None,
        images=None,
        rays=None,
        rays_noisy=None,
        t=None,
        mask=None,
        ndc_coordinates=None,
        unconditional_mask=None,
        compute_x0=False,
    ):
        """
        Args:
            images: (B, N, 3, H, W).
            t: (B,).
            rays: (B, N, 6, H, W).
            rays_noisy: (B, N, 6, H, W).
            ndc_coordinates: (B, N, 2, H, W).
            unconditional_mask: (B, N) or (B,). Should be 1 for unconditional samples
                and 0 else.
        """

        if features is None:
            features = self.feature_extractor(images, autoresize=False)

        B = features.shape[0]

        if unconditional_mask is not None and self.use_unconditional:
            null_token = self.null_token.reshape(1, 1, self.feature_dim, 1, 1)
            unconditional_mask = unconditional_mask.reshape(B, -1, 1, 1, 1)
            features = (
                features * (1 - unconditional_mask) + null_token * unconditional_mask
            )

        if isinstance(t, int) or isinstance(t, np.int64):
            t = torch.ones(1, dtype=int).to(features.device) * t
        else:
            t = t.reshape(B)

        if rays_noisy is None:
            rays_noisy, epsilon = self.forward_noise(rays, t, mask=mask)
        else:
            epsilon = None

        scene_features = torch.cat([features, rays_noisy], dim=2)
        if self.append_ndc:
            scene_features = torch.cat([scene_features, ndc_coordinates], dim=2)

        epsilon_pred = self.ray_predictor(scene_features, t)

        if compute_x0:
            t = t.reshape(-1, 1, 1, 1, 1)
            a = self.noise_scheduler.alphas_cumprod[t]
            x0 = (rays_noisy - torch.sqrt(1 - a) * epsilon_pred) / torch.sqrt(a)
            return epsilon_pred, x0
        return epsilon_pred, epsilon
    



class NoiseScheduler(nn.Module):
    def __init__(
        self,
        max_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        cos_power=2,
        num_inference_steps=100,
        type="linear",
    ):
        super().__init__()
        self.max_timesteps = max_timesteps
        self.num_inference_steps = num_inference_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.cos_power = cos_power
        self.type = type

        if type == "linear":
            self.register_linear_schedule()
        elif type == "cosine":
            self.register_cosine_schedule(cos_power)

        self.inference_timesteps = self.compute_inference_timesteps()

    def register_linear_schedule(self):
        self.register_buffer(
            "betas",
            torch.linspace(
                self.beta_start,
                self.beta_end,
                self.max_timesteps,
                dtype=torch.float32,
            ),
        )
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

    def register_cosine_schedule(self, cos_power, s=0.008):
        timesteps = (
            torch.arange(self.max_timesteps + 1, dtype=torch.float32)
            / self.max_timesteps
        )
        alpha_bars = (timesteps + s) / (1 + s) * np.pi / 2
        alpha_bars = torch.cos(alpha_bars).pow(cos_power)
        alpha_bars = alpha_bars / alpha_bars[0]
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

        self.register_buffer(
            "betas",
            betas,
        )
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

    def compute_inference_timesteps(
        self, num_inference_steps=None, num_train_steps=None
    ):
        # based on diffusers's scheduling code
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if num_train_steps is None:
            num_train_steps = self.max_timesteps
        step_ratio = num_train_steps // num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(int)
        )
        return timesteps

    def plot_schedule(self, return_image=False):
        fig = plt.figure(figsize=(6, 4), dpi=100)
        alpha_bars = self.alphas_cumprod.cpu().numpy()
        plt.plot(np.sqrt(alpha_bars))
        plt.grid()
        if self.type == "linear":
            plt.title(
                f"Linear (T={self.max_timesteps}, S={self.beta_start}, E={self.beta_end})"
            )
        else:
            self.type == "cosine"
            plt.title(f"Cosine (T={self.max_timesteps}, P={self.cos_power})")
        if return_image:
            image = plot_to_image(fig)
            plt.close(fig)
            return image
        

class SpatialDino(nn.Module):
    def __init__(
        self,
        freeze_weights=True,
        model_type="dinov2_vits14",
        num_patches_x=16,
        num_patches_y=16,
    ):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_type)
        self.feature_dim = self.model.embed_dim
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, autoresize=False):
        """
        Spatial dimensions of output will be H // 14, W // 14. If autoresize is True,
        then the output will be resized to the correct dimensions.

        Args:
            x (torch.Tensor): Images (B, C, H, W). Should be ImageNet normalized.
            autoresize (bool): Whether to resize the input to match the num_patch
                dimensions.

        Returns:
            feature_map (torch.tensor): (B, C, h, w)
        """
        *B, c, h, w = x.shape

        x = x.reshape(-1, c, h, w)

        # Output will be (B, H * W, C)
        features = self.model.forward_features(x)["x_norm_patchtokens"]
        features = features.permute(0, 2, 1)
        features = features.reshape(  # (B, C, H, W)
            -1, self.feature_dim, h // 14, w // 14
        )
        if autoresize:
            features = resize(features, size=(self.num_patches_y, self.num_patches_x))

        features = features.reshape(
            *B, self.feature_dim, self.num_patches_y, self.num_patches_x
        )
        return features
    
def resize(image, size=None, scale_factor=None):
    return nn.functional.interpolate(
        image,
        size=size,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=False,
    )

def plot_to_image(figure, dpi=100):
    """Converts matplotlib fig to a png for logging with tf.summary.image."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="raw", dpi=dpi)
    plt.close(figure)
    buffer.seek(0)
    image = np.reshape(
        np.frombuffer(buffer.getvalue(), dtype=np.uint8),
        newshape=(int(figure.bbox.bounds[3]), int(figure.bbox.bounds[2]), -1),
    )
    return image[..., :3]

# Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py

import math

import ipdb  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_xformers_attention=False,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if use_xformers_attention:
            attn = MEAttention
        else:
            attn = Attention
        self.attn = attn(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        in_channels=442,
        out_channels=6,
        width=16,
        hidden_size=1152,
        depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        max_num_images=8,
        P=1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.hidden_size = hidden_size
        self.max_num_images = max_num_images
        self.P = P

        self.x_embedder = PatchEmbed(
            img_size=self.width,
            patch_size=self.P,
            in_chans=in_channels,
            embed_dim=hidden_size,
            bias=True,
            flatten=False,
        )
        self.x_pos_enc = FeaturePositionalEncoding(
            max_num_images, hidden_size, width**2, P=self.P
        )
        self.t_embedder = TimestepEmbedder(hidden_size)

        try:
            import xformers

            use_xformers_attention = True
        except ImportError:
            # xformers not available
            use_xformers_attention = False

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    use_xformers_attention=use_xformers_attention,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, P, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)

        # print("unpatchify", c, p, h, w, x.shape)
        # assert h * w == x.shape[2]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nhpwqc", x)
        imgs = x.reshape(shape=(x.shape[0], h * p, h * p, c))
        return imgs

    def forward(self, x, t):
        """

        Args:
            x: Image/Ray features (B, N, C, H, W).
            t: Timesteps (N,).

        Returns:
            (B, N, D, H, W)
        """
        if isinstance(t, int):
            t = torch.tensor((t,)).to(x)
        B, N, c, h, w = x.shape
        P = self.P

        x = x.reshape((B * N, c, h, w))  # (B * N, C, H, W)
        x = self.x_embedder(x)  # (B * N, C, H / P, W / P)

        x = x.permute(0, 2, 3, 1)  # (B * N, H / P, W / P, C)
        # (B, N, H / P, W / P, C)
        x = x.reshape((B, N, h // P, w // P, self.hidden_size))
        x = self.x_pos_enc(x)  # (B, N, H * W / P ** 2, C)
        # TODO: fix positional encoding to work with (N, C, H, W) format.

        # Eval time, we get a scalar t
        if x.shape[0] != t.shape[0] and t.shape[0] == 1:
            t = t.repeat_interleave(B)

        t = self.t_embedder(t)

        for i, block in enumerate(self.blocks):
            x = x.reshape((B, N * h * w // P**2, self.hidden_size))
            x = block(x, t)  # (N, T, D)

        # (B, N * H * W / P ** 2, D)
        x = self.final_layer(
            x, t
        )  # (B, N * H * W / P ** 2,  6 * P ** 2) or (N, T, patch_size ** 2 * out_channels)

        x = x.reshape((B * N, w * w // P**2, self.out_channels * P**2))
        x = self.unpatchify(x)  # (B * N, H, W, C)
        x = x.reshape((B, N) + x.shape[1:])
        x = x.permute(0, 1, 4, 2, 3)  # (B, N, C, H, W)
        return x


class FeaturePositionalEncoding(nn.Module):
    def _get_sinusoid_encoding_table(self, n_position, d_hid, base):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(base, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def __init__(self, max_num_images=8, feature_dim=1152, num_patches=256, P=1):
        super().__init__()
        self.max_num_images = max_num_images
        self.feature_dim = feature_dim
        self.P = P
        self.num_patches = num_patches // self.P**2

        self.register_buffer(
            "image_pos_table",
            self._get_sinusoid_encoding_table(
                self.max_num_images, self.feature_dim, 10000
            ),
        )

        self.register_buffer(
            "token_pos_table",
            self._get_sinusoid_encoding_table(
                self.num_patches, self.feature_dim, 70007
            ),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_images = x.shape[1]

        x = x.reshape(batch_size, num_images, self.num_patches, self.feature_dim)

        # To encode image index
        pe1 = self.image_pos_table[:, :num_images].clone().detach()
        pe1 = pe1.reshape((1, num_images, 1, self.feature_dim))
        pe1 = pe1.repeat((batch_size, 1, self.num_patches, 1))

        # To encode patch index
        pe2 = self.token_pos_table.clone().detach()
        pe2 = pe2.reshape((1, 1, self.num_patches, self.feature_dim))
        pe2 = pe2.repeat((batch_size, num_images, 1, 1))

        x_pe = x + pe1 + pe2
        x_pe = x_pe.reshape(
            (batch_size, num_images * self.num_patches, self.feature_dim)
        )

        return x_pe
    
import torch.nn as nn
from xformers.ops import memory_efficient_attention


class MEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # MEA expects [B, N, H, D], whereas timm uses [B, H, N, D]
        x = memory_efficient_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=self.scale,
        )
        x = x.reshape(B, N, C)

        # Equivalent to doing the following:
        # q = q * self.scale
        # attn = q @ k.transpose(-2, -1)
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = attn @ v
        # x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x