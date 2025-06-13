"""
Generte Inverse Depth of SAM_1B using DepthAnythingV2
"""
import os
# Set the HF_ENDPOINT environment variable
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # before import all hf related pkgs
import json
import random
import tyro
import numpy as np
from tqdm.auto import tqdm
from PIL import Image, ImageFile, ImageOps
import einops
import ipdb  # noqa: F401
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go

import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from prometheus.datasets import Text2ImageDataset

def colorize_depth_maps(
    depth_map, min_depth = 0, max_depth = 1, cmap="Spectral_r", valid_mask=None
):
    """
    Colorize depth maps.
    Code borrow from Marigold
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().cpu().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def get_depth_for_dataset(
    dataset_root: str = "/nas6/yyb/MVImgNet",
    category: str = '0', # 'all'
    device : str = 'cuda:0'
):

    dataset = Text2ImageDataset(
        root_dir = '/nas6/yyb/SAM_1B/',
        csv_file = '/nas6/yyb/SAM_1B/metadata/tiny',
        resolution = 256,
        images_per_iter = 1,
        cfg_prob = 0.1,
        dataset_name = 'SAM1B')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=4, shuffle=False)
    # mean, std = np.array([0.485, 0.456, 0.406]),  np.array([0.229, 0.224, 0.225])
    disp_mean =torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
    disp_std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)


    # init depth-anything-V2
    model = AutoModelForDepthEstimation.from_pretrained("pretrained/huggingface/depth-anything/Depth-Anything-V2-Small-hf").to(device)

    for batch in tqdm(dataloader):
        images = batch['image'][:,0].to(device)
        #batch_size = images.shape[0]
        batch_size, C, H, W = images.shape
        with torch.no_grad():
            inputs = dict(pixel_values = ((images + 1)/2 - disp_mean) / disp_std)
            outputs = model(**inputs)
            pred_depth = outputs.predicted_depth
            pred_depth = F.interpolate(pred_depth[:,None], size=(H, W), align_corners=False, mode='bilinear')

        if True:
            max_ = pred_depth.reshape(pred_depth.shape[0], -1).max(dim = -1)[0]
            pred_depth = pred_depth / max_[:,None,None,None]
            colorize_depth = colorize_depth_maps(pred_depth, min_depth = 0, max_depth = 1,cmap="Spectral_r")
            colorize_depth_vis = einops.rearrange(colorize_depth, 'B C H W -> H (B W) C').cpu()
            image_vis =  einops.rearrange((images+1)/2, 'B C H W -> H (B W) C').cpu()
            image_vis = torch.cat((colorize_depth_vis, image_vis), dim = 0).clip(0,1).numpy()
            plt.imsave('aaa.png',image_vis)

        # for i in range(batch_size):
        #     depth_path = filepaths[i].replace('images', 'depths')
        #     os.makedirs(os.path.dirname(depth_path), exist_ok = True)
        #     image, bbox, depth  = images[i], bboxes[i], pred_depth[i]

        #     depth_to_save = depth[bbox[1]:bbox[3], bbox[0]:bbox[2]].cpu().numpy()
        #     depth_to_save = (255 * depth_to_save / depth_to_save.max()).astype(np.uint8)
        #     # depth = einops.rearrange(depth[:,bbox[1]:bbox[3], bbox[0]:bbox[2]].cpu().numpy(), 'C H W -> H W C')
        #     # image = std * image + mean
        #     depth_to_save = Image.fromarray(depth_to_save)
        #     depth_to_save.save(depth_path)
        #     print(depth_path)

if __name__ == "__main__":
     tyro.cli(get_depth_for_dataset)