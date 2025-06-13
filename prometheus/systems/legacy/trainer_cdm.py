from typing import Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import einops
import copy
import random
from diffusers import StableDiffusionPipeline, DDIMScheduler

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning
import warnings
warnings.filterwarnings("ignore")
import tqdm

from utils import import_str

from utils import matrix_to_square

from modules.renderers.gaussians_renderer import quaternion_to_matrix, matrix_to_quaternion

from modules.dit import *

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

class CDMBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cattn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sattn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True)

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate='tanh')
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, y, c):
        shift_ca, scale_ca, gate_ca, shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        x = x + gate_ca.unsqueeze(1) * self.cattn(modulate(self.norm1(x), shift_ca, scale_ca), y, y)
        x = x + gate_sa.unsqueeze(1) * self.sattn(modulate(self.norm2(x), shift_sa, scale_sa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x

class CDMModel(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        pipe = StableDiffusionPipeline.from_pretrained(
            self.opt.network.sd_model_key, local_files_only=self.opt.network.local_files_only
        )
        
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.requires_grad_(False)
        
        del pipe

        hidden_size = opt.network.cdm.hidden_size
        num_blocks = opt.network.cdm.num_blocks
        num_tokens = opt.network.cdm.num_tokens

        self.t_embedder = nn.Sequential(
            TimestepEmbedder(hidden_size),
            nn.SiLU(),
        )

        self.y_embedder = nn.Linear(1024, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_size))

        self.blocks = nn.ModuleList([CDMBlock(hidden_size,**self.opt.network.cdm.block_args) for i in range(num_blocks)])

        self.in_block = nn.Linear(4 + 3 + 4, hidden_size)
        self.out_block = nn.Linear(hidden_size, 4 + 3 + 4)

    @torch.no_grad()
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder[0].mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder[0].mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_block.weight, 0)
        nn.init.constant_(self.out_block.bias, 0)
    
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
        return self.y_embedder(text_embeddings)

    def forward(self, x, y, t):

        x = self.in_block(x) + self.pos_embed

        t = self.t_embedder(t)

        for block in self.blocks:
            x = block(x, y, t) 

        x = self.out_block(x)
        return x


class CDMSystem(LightningModule):
    '''trajectory generation system'''
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters(opt)
        self.opt = opt
        
        self.model = CDMModel(opt)

        self.scheduler = DDIMScheduler(beta_schedule='scaled_linear', beta_start=0.00085, beta_end=0.012, prediction_type="sample", clip_sample=False, steps_offset=9, rescale_betas_zero_snr=True, set_alpha_to_one=True)

        self.register_buffer("alphas_cumprod", self.scheduler.alphas_cumprod, persistent=False)
     
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = 0
        self.max_step = int(self.num_train_timesteps)

        self.model_ema = copy.deepcopy(self.model).requires_grad_(False)
        
    def configure_optimizers(self):
        params = []
        for p in self.model.parameters():
            if p.requires_grad: params.append(p)
        optimizer = torch.optim.AdamW(params, lr=self.opt.training_cdm.learning_rate / self.opt.training_cdm.accumulate_grad_batches, weight_decay=self.opt.training_cdm.weight_decay, betas=self.opt.training_cdm.betas)
        return optimizer
    
    def add_noise(self, x, noise, t):
        x_noisy = self.scheduler.add_noise(x, noise, t)
        return x_noisy

    def camera_to_token(self, cameras):
        B, N, _ = cameras.shape

        RT = cameras[:, :, :12].reshape(B, N, 3, 4)
        # rotation
        rotation = matrix_to_quaternion(RT[:, :, :, :3])
        # translation
        translation = RT[:, :, :, 3]
        # fx, fy, cx, cy
        intrinsics = torch.stack([cameras[:, :, 12] / cameras[:, :, 16], 
                                 cameras[:, :, 13] / cameras[:, :, 17], 
                                 cameras[:, :, 14] / cameras[:, :, 16], 
                                 cameras[:, :, 15] / cameras[:, :, 17]], dim=2)

        return torch.cat([rotation, translation, intrinsics], dim=2)

    def token_to_camera(self, tokens, image_size):
        B, N, _ = tokens.shape

        R = quaternion_to_matrix(tokens[:, :, :4]) # B, N, 3, 3
        T = tokens[:, :, 4:7].reshape(B, N, 3, 1) # B, N, 3, 1

        RT = torch.cat([R, T], dim=3).reshape(B, N, 12)

        intrinsics = torch.stack([tokens[:, :, 7] * image_size, 
                                  tokens[:, :, 8] * image_size, 
                                  tokens[:, :, 9] * image_size, 
                                  tokens[:, :, 10] * image_size,
                                  torch.full((B, N), fill_value=image_size, device=self.device),
                                  torch.full((B, N), fill_value=image_size, device=self.device),
                                 ], dim=2)

        return torch.cat([RT, intrinsics], dim=2)

    def training_step(self, batch, _):
        update_average(self.model_ema, self.model)
        
        cameras, texts = batch

        B, N, _ = cameras.shape

        with torch.no_grad():
            tokens = self.camera_to_token(cameras)
            t = torch.randint(0, self.num_train_timesteps, (B,), dtype=torch.long, device=self.device)
            tokens_noisy = self.add_noise(tokens, torch.randn_like(tokens), t)
            text_embeddings = self.model.encode_text(texts)

        tokens_pred = self.model(tokens_noisy, text_embeddings, t)
        _tokens_pred = tokens_pred.clone()

        # 
        _tokens_pred[:, :, :4] = F.normalize(tokens_pred[:, :, :4], dim=-1)

        loss = F.mse_loss(_tokens_pred, tokens)

        self.log('losses/loss', loss, sync_dist=True)
        # self.log('losses/loss_ext', loss_ext, sync_dist=True)
            
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        return 

    # @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def inference(self, text, num_inference_steps=100, guidance_scale=7.5, image_size=512, return_each=False):
        B = 1
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = self.scheduler.timesteps

        tokens_noisy = torch.randn(B, self.opt.network.cdm.num_tokens, 4 + 3 + 4, device=self.device)

        text_embeddings = self.model.encode_text([text])
        # uncond_text_embeddings =  self.model.encode_text(['']).repeat(B, 1, 1)
        # text_embeddings = torch.cat([text_embeddings, uncond_text_embeddings], 0)

        results = []

        for i, t in tqdm.tqdm(enumerate(timesteps), total=len(timesteps)):
            t = t[None].repeat(B)

            tokens_pred = self.model(tokens_noisy, text_embeddings, t)

            tokens_pred[:, :, :4] = F.normalize(tokens_pred[:, :, :4], dim=-1)

            results += [tokens_pred]
            
            tokens_noisy = self.scheduler.step(tokens_pred.cpu(), t.cpu(), tokens_noisy.cpu(), eta=0).prev_sample.to(self.device)

        if return_each:
            return [self.token_to_camera(result, image_size=image_size) for result in results]
        else:
            return self.token_to_camera(tokens_noisy, image_size=image_size)

def parse_resume_path():
    pass


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dual3dgs_mvimgnet_objaverse_laion.yaml", help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    try:
        num_nodes = int(os.environ["NUM_NODES"])
    except:
        os.environ["NUM_NODES"] = '1'
        num_nodes = 1
    
    print(num_nodes)

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    train_dataset = import_str(opt['dataset_cdm']['module'])(**opt['dataset_cdm']['args'], fake_length=1000 * opt.training_cdm.batch_size * len(opt.training_cdm.gpus) * opt.training_cdm.accumulate_grad_batches * num_nodes)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.training_cdm.batch_size, num_workers=opt.training_cdm.num_workers, shuffle=False)
    
    val_dataset = import_str(opt['dataset_cdm']['module'])(**opt['dataset_cdm']['args'], fake_length=4 * len(opt.training_cdm.gpus) * num_nodes)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=1, shuffle=False)

    system = CDMSystem(opt)

    trainer = Trainer(
            default_root_dir=f"logs/{opt.experiment_name}",
            max_steps=opt.training_cdm.max_steps,
            check_val_every_n_epoch=opt.training_cdm.check_val_every_n_epoch,
            log_every_n_steps=1,
            accumulate_grad_batches=opt.training_cdm.accumulate_grad_batches,
            precision=opt.training_cdm.precision,
            accelerator='gpu',
            gpus=opt.training_cdm.gpus,
            strategy=DDPPlugin(find_unused_parameters=False)
                               if len(opt.training_cdm.gpus) > 1 else None,
            benchmark=True,
            gradient_clip_val=opt.training_cdm.gradient_clip_val,
            resume_from_checkpoint=opt.training_cdm.resume_from_checkpoint,
            # track_grad_norm=1,
            # detect_anomaly=True,
            num_nodes=num_nodes,
        )
    
    trainer.fit(model=system, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        