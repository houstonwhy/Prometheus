"""Script for visualizing Large Scale Text2Image Dataset (e.g., SAM1B) on Ant cluster."""

import sys
import os
import tqdm
import json
import numpy as np
# import tyro
import torch
# from pathlib import Path
# from easydict import EasyDict
# from omegaconf import OmegaConf
# import pickle as pkl
# from prometheus.utils.image_utils import postprocess_image
from tqdm import tqdm
import lightning
from prometheus.datasets import *
from einops import rearrange
from prometheus.utils import import_str
try:
    from pcache_fileio import fileio
except ImportError:
    print('pcache_fileio module not found.')
# import plotly
import hydra
from omegaconf import DictConfig, OmegaConf
# from omegaconf.omegaconf import open_dict
# from dataclasses import dataclass

# from prometheus.utils.visualization import plotly_scene_visualization


# @dataclass
# class Args:
#     """Command-line arguments."""
#     config: str = ""


@hydra.main(
    version_base=None,
    config_path="configurations",
    config_name="config",
)
def run(cfg: DictConfig) -> None:
    """Main function to run the visualization script."""
    # Load the dataset configuration
    # args = tyro.cli(Args)
    # dataset_cfg = EasyDict(OmegaConf.load(f'configurations/dataset/mvldm_dataset.yaml'))
    # cfg['dataset']['train'].update(dataset_cfg['train'])

    # Set the random seed for reproducibility
    lightning.seed_everything(cfg.seed)

    dataset = import_str(cfg['dataset']['train']['module'])(
        **cfg['dataset']['train']['args'],
        fake_length=1000 * cfg.experiment.training.batch_size,
        debug=False
    )
        
    # dataset = import_str(cfg['dataset']['train']['args']['datasets'][0]['module'])(
    #     **cfg['dataset']['train']['args']['datasets'][0]['args'],
    #     fake_length=1000 * cfg.experiment.training.batch_size,
    #     debug=False
    # )

    # Create a DataLoader for the dataset
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.experiment.training.batch_size,
        num_workers=12,
        shuffle=True,
        # pin_memory=True,
        # persistent_workers=True
    )

    # Iterate over the DataLoader
    # 'da4d13075753918a'
    # idx_ = [i for i, idx in enumerate(dataset.datasets[0].datasets[3].items) if idx == 'da4d13075753918a'][0]
    # dataset.datasets[0].datasets[3].get_raw_data(idx_)
    # dataset[0].datasets
    for batch in tqdm(data_loader):
        pass
        # print('Processing batch...')
    print('done')
        # Add your visualization code here
        # For example:
        # images = postprocess_image(rearrange(batch['images_mv'][0, :num_input_view], 'N C H W -> C H (N W)')[None], return_PIL=True)[0]
        # cam_vis = plotly_scene_visualization(batch['cameras_mv'][0, :num_input_view], img_return=True).save(f"{datast_preview_path}/{batch['scene_name_mv'][0].split('/')[-1]}_cam.png")
        # images.save(f"{datast_preview_path}/{batch['scene_name_mv'][0].split('/')[-1]}.png")


if __name__ == "__main__":
    run()