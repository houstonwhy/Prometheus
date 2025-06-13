import os
import torch
import pickle as pkl
import pandas as pd

# todo pkl 读取/ssd0/yyb/PlatonicGen/benchmark/scene_benckmark 里的所有文件
decoder_ckpt_path = 'pretrained/gsdecoder.ckpt'
mvldm_ckpt_path = 'outputs/ckpts/prometheus_k8s_mvldm_dir3d_exp_mvldm_dataset_mvldm_viewcond_mvldm_dir3d_exp_exp12d_emanormfull/epoch=153-step=154000.ckpt'
final_ckpt_path = 'pretrained/full.ckpt'

params_traj = torch.load('pretrained/model.ckpt', weights_only=False, map_location='cpu')
params_decoder = torch.load(decoder_ckpt_path, weights_only=False, map_location='cpu')

# delete keys
params_decoder.pop('lr_schedulers')
params_decoder.pop('optimizer_states')
params_decoder.pop('loops')
params_decoder.pop('epoch')
params_decoder.pop('global_step')
params_decoder.pop('callbacks')
# del ema model
for key in dict(params_decoder['state_dict']):
    if 'ema' in key:
        params_decoder['state_dict'].pop(key)


params_mvldm = torch.load(mvldm_ckpt_path, weights_only=False, map_location='cpu')

# delete keys
params_mvldm.pop('lr_schedulers')
params_mvldm.pop('optimizer_states')
params_mvldm.pop('loops')
params_mvldm.pop('epoch')
params_mvldm.pop('global_step')
params_mvldm.pop('callbacks')
# del ema model
for key in dict(params_mvldm['state_dict']):
    if 'model_ema' in key:
        params_mvldm['state_dict'].pop(key)


final_ckpt = {  
    'traj': params_traj['traj_dit'],
    'decoder': params_decoder,
    'mvldm': params_mvldm,
}


torch.save(final_ckpt, final_ckpt_path)











# todo逐一写入csv