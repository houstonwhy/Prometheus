"""
Base System
"""
#pyling:disable=import-error
import os
import io
import copy
import re
from easydict import EasyDict
import numpy as np
from pathlib import Path
import random
# import tqdm
import lpips
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
from lightning import LightningModule

# from diffusers import DDIMScheduler
from einops import rearrange, repeat
from prometheus.utils import sample_rays, embed_rays
from prometheus.models import GSDecoderModel
from prometheus.utils.image_utils import postprocess_image, colorize_depth_maps
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from .depth_loss import ScaleAndShiftInvariantLoss, disp_to_depth
from lightning.pytorch.callbacks import ModelCheckpoint
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_ddim  import DDIMScheduler
from diffusers.schedulers.scheduling_edm_euler  import EDMEulerScheduler
from diffusers.schedulers.scheduling_edm_dpmsolver_multistep import EDMDPMSolverMultistepScheduler
from prometheus.utils import import_str

class BaseSystem(LightningModule):
    '''Base Training system'''
    def __init__(self, opt, mode = 'training'):
        super().__init__()
        self.save_hyperparameters(opt)
        self.mode = mode
        self.opt = opt
        self.configure_job_name()
        # self.image_size = self.opt.network.image_size
        self.latent_size = self.opt.experiment.image_size // 8
        self.latent_channel = self.opt.network.latent_channel
        if mode == 'training':
            self.existing_checkpoint_path = self.parse_jobname(self.opt.training.resume_from_checkpoint)
        else:
            self.existing_checkpoint_path = False
        self.model = import_str(self.opt.algorithm.module)(opt.network, mode = mode)
        
        # if mode == 'training' and opt.training.resume_from_director3d:
        #     self.load_from_director3d_ckpts()

        self.configure_lpips_fn()
        self.psnr_fn = PSNR()
        # if True:
        #     # self.configure_lpips_fn()
        if mode == 'trainin':
            self.configure_depth_fn()
        self.high_noise_level = opt.training.get('high_noise_level', False)
        self.configure_noise_scheduler()
        self.model_ema = copy.deepcopy(self.model).requires_grad_(False)
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = 0
        self.max_step = int(self.num_train_timesteps)
        self.log_every_n_step = self.opt.training.log_every_n_step
        self.num_input_views = self.opt.training.num_input_views
        self.num_novel_views = self.opt.training.num_novel_views
        # single-vie-data num per iteration
        self.single_view_num = self.opt.training.single_view_num
        
        self.render_bg_color = opt.training.get('render_bg_color', 'random')
        # set up decoder
        self.decoder_mode = 'RGB' # decoder input type
        if self.model.extra_latent_channel == 0:
            self.decoder_mode = 'RGB'
        elif self.model.extra_latent_channel == 4:
            self.decoder_mode = 'RGBD'
        elif self.model.extra_latent_channel == 6:
            self.decoder_mode = 'RGBPose'
        elif self.model.extra_latent_channel == 10:
            self.decoder_mode = 'RGBDPose'
        elif self.model.extra_latent_channel == 11:
            self.decoder_mode = 'RGBDPoseM'
        else:
            raise ValueError(f'Unsupport decoder_mode with {self.model.extra_latent_channel} extra latent channel')
        
    def configure_job_name(self):
        now = datetime.now()
        formatted_time = now.strftime("%Y%m%d_%H%M")
        self.job_time = formatted_time
        self.job_name = f"{self.opt.name}_{self.opt.experiment._name}_{self.opt.dataset._name}_{self.opt.algorithm._name}_{self.opt.experiment._name}_{self.opt.tags}" 
        self.job_name_with_t = self.job_name + f'_{self.job_time}'
        self.ckpt_base_dir = os.path.join(self.opt.output_dir, 'ckpts')
        self.output_dir = self.opt.output_dir
    
    
    def configure_checkpoint_callback(self, add_time_step = True):
        if add_time_step:
            dirpath = os.path.join(self.ckpt_base_dir, self.job_name_with_t)
        else:
            dirpath = os.path.join(self.ckpt_base_dir, self.job_name)
        # os.makedirs(dirpath, exist_ok=True)
        return ModelCheckpoint(
            dirpath= dirpath,        # Path where checkpoints will be saved
            #filename='{epoch}',        # Filename for the checkpoints
            #save_top_k=5,             # Set to -1 to save all checkpoints
            every_n_epochs=1,          # Save a checkpoint every K epochs
            # every_n_train_steps=200,
            #filename='epoch:{epoch}_psnr:{val/PSNR_mv}',
            #monitor='{val/PSNR_mv}',
            verbose=True,
            save_on_train_epoch_end=True,  # Ensure it saves at the end of an epoch, not the beginning
        )

    def parse_jobname(self, dir_name = 'latest'):

        if (not dir_name) or (not os.path.exists(self.ckpt_base_dir)):
            return None
        if dir_name == 'latest':
            def get_step_num(file_path):
                return int(re.search(r'step=(\d+)', Path(file_path).name).group(1))
            ckpt_list = []
            for dd in Path(self.ckpt_base_dir).iterdir():
                if self.job_name in dd.name:
                    ckpt_list += dd.glob('*.ckpt')
            if len(ckpt_list) == 0:
                return None
            latest_idx = np.argmax([get_step_num(ff) for ff in ckpt_list])
            ckpt_path = ckpt_list[latest_idx]
            resume_job_name = ckpt_path.parent.name
            timestep_pattern = re.compile(r'.*\d{4}\d{2}\d{2}_\d{2}\d{2}$')
            if not timestep_pattern.match(resume_job_name):
                resume_job_name = self.job_name_with_t
                print(f'Choose as {ckpt_path} latest checkpoint, start new job job_id:{resume_job_name}')
            else:
                 print(f'Choose as {ckpt_path} latest checkpoint, resume from wandb job_id:{resume_job_name}')
        else:
            ckpt_path = os.path.join(self.ckpt_base_dir, dir_name)
            resume_job_name = self.job_name_with_t
            if os.path.isdir(ckpt_path):
                ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
            else:
                assert os.path.exists(ckpt_path)
       
        return ckpt_path
    
    
    def configure_optimizers(self):
        """"""
        params = []
        for p in self.model.parameters():
            if p.requires_grad: params.append(p)
        optimizer = torch.optim.AdamW(params, lr=self.opt.training.learning_rate / self.opt.training.accumulate_grad_batches, weight_decay=self.opt.training.weight_decay, betas=self.opt.training.betas)
        return optimizer

    def configure_lpips_fn(self):
        self.lpips_fn = lpips.LPIPS(net='vgg', version='0.1', model_path=self.opt.training.lpips_model_path).eval().requires_grad_(False)

    def configure_depth_fn(self):
        self.disp_fn = AutoModelForDepthEstimation.from_pretrained(self.opt.training.depth_model_path) # works well
        #self.disp_fn = pipeline(task="depth-estimation", model="pretrained/huggingface/depth-anything/Depth-Anything-V2-Small-hf").model.eval() # doesn't work
        self.register_buffer("disp_image_mean", torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer("disp_image_std", torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), persistent=False)
        self.disp_loss =  ScaleAndShiftInvariantLoss()

    def configure_noise_scheduler(self, sigma_data_ = -1):
        """_summary_

        _extended_summary_
        """
        scheduler_type = self.opt.experiment.get('scheduler_type', 'edm')
        timestep_type = self.opt.experiment.get('timestep_type', 'discrete')
        # scheduler_type = 'ddim'
        if scheduler_type == 'ddim':
                # x0 predeicting, load form sd21 base ckpt
                scheduler = DDIMScheduler(
                    beta_schedule='sample', 
                    beta_start=0.00085, 
                    beta_end=0.012, 
                    prediction_type=prediction_type, 
                    clip_sample=False, 
                    steps_offset=9, 
                    rescale_betas_zero_snr=True, 
                    set_alpha_to_one=True
                    )
                # sample  step -> sigma
                #TODO rewrite DDIM in EDM's formulation
                self.register_buffer("alphas_cumprod", scheduler.alphas_cumprod, persistent=False)

        elif scheduler_type == 'eprediction':
            scheduler_vars = EasyDict(
                prediction_type = "epsilon",
                num_train_timesteps=1000,
                ode_solver = 'euler',
                c_skip = lambda sigma: 1  / (sigma ** 2 + 1),
                c_out = lambda sigma: sigma / ((sigma ** 2 + 1)** 0.5),
                c_in = lambda sigma: 1 / ((1 + sigma ** 2)** 0.5),
                c_noise = lambda sigma: sigma.log() / 4,
                weight = lambda sigma: (sigma ** 2 + 1) / (sigma ** 2)
                )

            scheduler = EulerAncestralDiscreteScheduler(
                    beta_schedule='scaled_linear', 
                    beta_start=0.00085, 
                    beta_end=0.012, 
                    prediction_type='epsilon', 
                    # clip_sample=False, 
                    # steps_offset=9, 
                    rescale_betas_zero_snr=True, 
                    # set_alpha_to_one=False
                    )
            self.sigmas = scheduler.sigmas.to(self.device)

        elif scheduler_type == 'vprediction':
            scheduler_vars = EasyDict(
                timestep_type = timestep_type,
                prediction_type = "vprediction",
                num_train_timesteps=1000,
                ode_solver = 'euler',
                c_skip = lambda sigma: 1  / (sigma ** 2 + 1),
                c_out = lambda sigma: -sigma / ((sigma ** 2 + 1)** 0.5),
                c_in = lambda sigma: 1 / ((1 + sigma ** 2)** 0.5),
                c_noise = lambda sigma: sigma.log() * 0.25,
                weight = lambda sigma: (sigma ** 2 + 1) / (sigma ** 2)
                )

                # v predeicting, load form sd21-v768 ckpt
                # Scheduler type borrow from Zero123++
            scheduler = EulerDiscreteScheduler(
            # scheduler = EulerAncestralDiscreteScheduler(
                    beta_start=0.00085, 
                    beta_end=0.012, 
                    prediction_type='v_prediction', 
                    # clip_sample=False, 
                    steps_offset=1, 
                    beta_schedule='linear', 
                    rescale_betas_zero_snr=True, 
                    # skip_prk_steps=True,
                    timestep_type = timestep_type,
                    # timestep_type = scheduler_vars.timestep_type,
                    timestep_spacing="linspace",
                    # set_alpha_to_one=False
                    )
                # sample  step -> sigma
                #TODO rewrite DDIM in EDM's formulation
            # self.register_buffer("alphas_cumprod", scheduler.alphas_cumprod, persistent=False)
            self.timesteps_to_sigmas = torch.flip(scheduler.sigmas[:-1], dims=[0]).cpu()
            if timestep_type == "continuous":
                self.timesteps_to_tconds = self.timesteps_to_sigmas.log() * 0.25
            elif timestep_type == "discrete":
                self.timesteps_to_tconds = torch.range(0, 1000-1)
            else:
                raise ValueError
            # if scheduler_vars.timestep_type == "continuous":
            #     self.t_to_tconds = torch.cat([scheduler.sigmas[(scheduler.timesteps == t).nonzero()] for t in range(1000)])[:,0].cpu()
            # elif scheduler_vars.timestep_type == "discrete":
            #     self.t_to_tconds = torch.cat([scheduler.sigmas[(scheduler.timesteps == t).nonzero()] for t in range(1000)])[:,0].cpu()
        

        
        elif scheduler_type == 'edm':
            # scheduler kwargs borrow from -> https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/scheduler/scheduler_config.json
            if not self.high_noise_level:
                #naive edm
                sigma_data = 1.0 if sigma_data_ == -1 else sigma_data_
                sigma_min = 0.002
                sigma_max = 80
                sigma_schedule="karras"
                p_mean = -0.5
                # p_std, p_mean = 1.2, -0.5
                p_std = 1.2
            else:
                # follow CAT3D shift logSNR  by log(8)
                sigma_min = 0.002
                sigma_max = 10000
                sigma_schedule="exponential"
                p_mean = 1.5
                p_std = 2.0
                sigma_data = 1.0 if sigma_data_ == -1 else sigma_data_

            scheduler_vars = EasyDict(
            # borrow from Tab1 of edm paperhttps://arxiv.org/pdf/2206.00364
            # sample sigma during training ln(\sigma) ~ N(p_mean, (P_std)^2))
            sigma_min = sigma_min,
            sigma_max = sigma_max,
            sigma_schedule = sigma_schedule,
            sigma_data = sigma_data,
            p_mean = p_mean,
            p_std = p_std,
            rho = 7.0,
            prediction_type = "epsilon",
            num_train_timesteps=1000,
            ode_solver = 'euler',
            c_skip = lambda sigma: sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2),
            c_out = lambda sigma: sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt(),
            c_in = lambda sigma: 1 / (sigma_data ** 2 + sigma ** 2).sqrt(),
            c_noise = lambda sigma: sigma.log() / 4,
            weight = lambda sigma: (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
            )
            if scheduler_vars.ode_solver == 'euler':
                scheduler = EDMEulerScheduler(
                sigma_min=scheduler_vars.sigma_min,
                sigma_max=scheduler_vars.sigma_max,
                sigma_data=scheduler_vars.sigma_data,
                sigma_schedule=scheduler_vars.sigma_schedule,
                num_train_timesteps=scheduler_vars.num_train_timesteps,
                prediction_type=scheduler_vars.prediction_type,
                rho=scheduler_vars.rho,
                )
            elif scheduler_vars.ode_solver == 'dpmsolver++':
                scheduler = EDMDPMSolverMultistepScheduler(
                sigma_min=scheduler_vars.sigma_min,
                sigma_max=scheduler_vars.sigma_max,
                sigma_data=scheduler_vars.sigma_data,
                sigma_schedule=scheduler_vars.num_train_timesteps,
                num_train_timesteps=scheduler_vars.num_train_timesteps,
                prediction_type=scheduler_vars.prediction_type,
                rho=scheduler_vars.rho,
                )
            else:
                raise ValueError(f'Unsupport ODE solver {scheduler_vars.ode_solver} for {scheduler_type}')
                
        else:
            raise ValueError(f'Unsupport DDM training scheme {scheduler_type}')

        self.scheduler = scheduler
        self.scheduler_type = scheduler_type
        self.scheduler_vars= scheduler_vars

    def load_from_director3d_ckpts(self):
        if self.opt.debug:
            with open(self.opt.training.resume_from_director3d, 'rb') as f:
                os.fsync(f.fileno())
                director3d_ckpt = torch.load(f, weights_only=False, map_location='cpu')['gm_ldm']
        else:
            director3d_ckpt = torch.load(self.opt.training.resume_from_director3d, weights_only=False, map_location='cpu')['gm_ldm']
        # del ckpt_buffer
        # Load MVDiff
        if hasattr(self.model, 'unet'):
            self.model.unet.load_state_dict(director3d_ckpt, strict=False)
        if False:
        # Load GS-Decoder
            decoder_state_dict = {}
            in_channels = self.model.vae.decoder.conv_in.weight.shape[1]
            for key, value in director3d_ckpt.items():
                if 'vae.decoder.' in key:
                    decoder_state_dict[key.replace('vae.decoder.', '')] = value
            decoder_state_dict['conv_in.weight'] = decoder_state_dict['conv_in.weight'][:,:in_channels,:,:]
            self.model.vae.decoder.load_state_dict(decoder_state_dict,strict=False)
            del director3d_ckpt
            print(f"Resume GM-LDM form pretrained Director 3D ckpt {self.opt.training.resume_from_director3d}. ")

   
    def configure_refiner(self):
        self.refiner = None

    @torch.no_grad()
    def get_depth_gt(self, x, return_disp = False, normalize = True):
        B, N, C, H, W = x.shape
        x = rearrange(x, 'B N C H W -> (B N) C H W')
        #x = F.interpolate(x, size=(518, 518), align_corners=False, mode='bicubic')
        inputs = dict(pixel_values =((x + 1)/2 - self.disp_image_mean) / self.disp_image_std)
        disps = self.disp_fn(**inputs).predicted_depth.unsqueeze(1)
        disps = F.interpolate(disps, size=(H, W), align_corners=False, mode='bilinear')
        if normalize and return_disp:
            disps_flatten = disps.flatten(1, -1)
            min_disps_flatten = disps_flatten.min(dim=1)[0].reshape(B * N, 1, 1, 1)
            max_disps_flatten = disps_flatten.max(dim=1)[0].reshape(B * N, 1, 1, 1)
            disps = (disps - min_disps_flatten) / (max_disps_flatten - min_disps_flatten)
            disps = disps.clip(0,1)
            
        disps = rearrange(disps, '(B N) C H W -> B N C H W', B = B, N = N)
        if return_disp:
            return disps
        else:
            return disp_to_depth(disps, normalize = normalize)
        
    # @torch.amp.autocast("cuda", enabled=True) # -> fp16 by default!!!!
    def training_step(self, batch):
        """XX"""
        raise NotImplementedError('?')

    @torch.no_grad()
    def validation_step(self, batch):
        """XX"""
        raise NotImplementedError('?')

    def inference_one_step(self):
        """"""
        raise NotImplementedError('?')

    
    def add_noise(self, x, noise, t):
        """t: timestep / sigma"""
        re_flag = False
        if len(x.shape) == 5:
            B, N, C ,H, W = x.shape
            x = rearrange(x, 'B N C H W -> (B N) C H W')
            noise = rearrange(noise, 'B N C H W -> (B N) C H W')
            re_flag = True
        
        t = t.reshape(-1,1,1,1)
        assert t.shape[0] == x.shape[0]

        x_noisy = self.scheduler.add_noise(x, noise, t)

        if re_flag:
            x_noisy = rearrange(x_noisy, '(B N) C H W -> B N C H W', B = B, N = N)

        return x_noisy

    def rand_log_normal(self, shape, dtype = None, device = None, loc=0., scale=1.):
        """Draws samples from an lognormal distribution."""
        if not dtype:
            dtype = self.dtype
        if not device:
            device = self.device
        u = torch.rand(shape, dtype=dtype, device=self.device) * (1 - 2e-7) + 1e-7
        return torch.distributions.Normal(loc, scale).icdf(u).exp()

    @torch.amp.autocast("cuda", enabled=False)
    def prepare_data_for_different_task(self, texts, cameras, task='text_to_3d'):
        """"""
        B, N, _ = cameras.shape

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
                pass
        return texts, cameras

    @staticmethod
    @torch.no_grad()
    def update_average(model_tgt, model_src, beta=0.995):
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

    
    def forward_single_view(self, batch, mode = 'val', batch_idx=0):
        raise NotImplementedError
    
    def forward_multi_view(self, batch, mode = 'val', batch_idx=0):
        raise NotImplementedError