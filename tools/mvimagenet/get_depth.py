"""
Generte Inverse Depth of MVImgNet using DepthAnythingV2
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
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
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


def pad_to_square(image):
    """
    Pads the input image to a square shape by adding equal padding on both sides of the shorter dimension.

    Args:
    image (PIL.Image): The input image to be padded.

    Returns:
    tuple: A tuple containing the padded image and the bounding box of the original image within the padded image.
    """
    # Get the original width and height of the image
    width, height = image.size

    # Calculate the maximum side length and the required padding
    max_side = max(width, height)
    delta_w = max_side - width
    delta_h = max_side - height

    # Calculate the padding to be added on each side
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

    # Expand the image with the calculated padding, filling with black color
    new_image = ImageOps.expand(image, padding, fill=(0, 0, 0))

    # Calculate the bounding box of the original image within the new image
    bbox = np.array((padding[0], padding[1], padding[0] + width, padding[1] + height))

    return new_image, bbox

class MVImgNetDatasetForLabeling(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 num_input_views=8,
                 category = '0',
                #  num_novel_views=8,
                 sample_rate=1,
                 img_size=504, # 504 = 36 * 14
                 max_num_scenes=-1,
                 fake_length=None,
                 normalized_cameras=False,
                 use_caption=False,
                 drop_text_p=0,
                 idfile_path=None
                 ):
        super().__init__()

        basedirs = []

        category_list = []
        if category == 'all':
            category_list = os.listdir(path)
        else:
            category_list = [category]
        for cate in category_list:
            if not os.path.isdir(os.path.join(path, cate)):
                continue
            for id in os.listdir(os.path.join(path, cate)):
                if not os.path.isdir(os.path.join(path, cate, id)):
                    continue
                basedirs.append(os.path.join(path, cate, id))

        print(f'categoty {category} cpontains {len(basedirs)} scenes')
        # basedirs = basedirs[:10]

        # def filter_dataset(basedir):
        #     try:
        #         filenames = [f for f in sorted(os.listdir(os.path.join(basedir, 'images'))) if f.endswith('jpg')]
        #         if len(filenames) < (num_input_views - 1) * sample_rate + 1:
        #             return False
        #         if not os.path.exists(os.path.join(basedir, 'cameras.npz')):
        #             return False
        #         # if len(os.listdir(os.path.join(basedir, 'cameras'))) != len(filenames) :
        #         #     return False
        #         return True
        #     except:
        #         return False

        #basedirs = list(filter(filter_dataset, basedirs))

        basedirs = basedirs if max_num_scenes < 0 else basedirs[:max_num_scenes]

        # if idfile_path is not None:
        #     with open(idfile_path, 'r') as f:
        #         scenes_ids = f.readlines()
        #     scenes_ids = [scene[:-1] for scene in scenes_ids]
        #     scenes_list = list(set(scenes_list) & set(scenes_ids))

        self.basedirs = basedirs
        self.img_size = img_size
        filenames = []
        for basedir in basedirs:
            filenames += [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) if f.endswith('jpg')]
        self.filenames = filenames

        self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(self.img_size, antialias=True),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        print(f'MVImgNet Dataset Length: {len(basedirs)}, frames: {len(filenames)}')


    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, index):

        file_name = self.filenames[index]
        image = Image.open(file_name).convert("RGB")
        image, bbox = pad_to_square(image)
        image_504 = self.transform(image) # 504 = 14 * 36
        scale = image.size[1] / self.img_size
        bbox = np.round(bbox / scale).astype(np.int32)

        data = {
            'valid_bbox': bbox,
            'filepath':file_name,
            'images':image_504
                }
        return data

def get_depth_for_dataset(
    dataset_root: str = "/nas6/yyb/MVImgNet",
    category: str = '0', # 'all'
    device : str = 'cuda:0'
):
    # dataset_root = "/nas6/yyb/MVImgNet"
    # category = '0' # 'all'
    # device = 'cuda:0'

    dataset = MVImgNetDatasetForLabeling(
        path = dataset_root,
        category = category,
        normalized_cameras = True,
        fake_length=1,
        use_caption = False,
        drop_text_p = 0.1,
        max_num_scenes = -1,
        num_input_views =8,
        sample_rate = 4,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=4, shuffle=False)
    mean, std = np.array([0.485, 0.456, 0.406]),  np.array([0.229, 0.224, 0.225])


    # init depth-anything-V2
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device)

    for batch in tqdm(dataloader):
        images, bboxes, filepaths = \
            batch['images'].to(device), batch['valid_bbox'].to(device), batch['filepath']
        batch_size = images.shape[0]

        with torch.no_grad():
            inputs = dict(pixel_values = images)
            outputs = model(**inputs)
            pred_depth = outputs.predicted_depth

        # if False:
        #     max_ = pred_depth.reshape(pred_depth.shape[0], -1).max(dim = -1)[0]
        #     pred_depth = pred_depth[:,None] / max_[:,None,None,None]
        #     colorize_depth = colorize_depth_maps(pred_depth, 0, 1)
        #     colorize_depth_vis = einops.rearrange(colorize_depth, 'B C H W -> H (B W) C').cpu()
        #     image_vis =  einops.rearrange(images, 'B C H W -> H (B W) C').cpu()
        #     image_vis = torch.cat((colorize_depth_vis, image_vis), dim = 0).clip(0,1).numpy()
        #     plt.imsave('aaa.png',image_vis)

        for i in range(batch_size):
            depth_path = filepaths[i].replace('images', 'depths')
            os.makedirs(os.path.dirname(depth_path), exist_ok = True)
            image, bbox, depth  = images[i], bboxes[i], pred_depth[i]

            depth_to_save = depth[bbox[1]:bbox[3], bbox[0]:bbox[2]].cpu().numpy()
            depth_to_save = (255 * depth_to_save / depth_to_save.max()).astype(np.uint8)
            # depth = einops.rearrange(depth[:,bbox[1]:bbox[3], bbox[0]:bbox[2]].cpu().numpy(), 'C H W -> H W C')
            # image = std * image + mean
            depth_to_save = Image.fromarray(depth_to_save)
            depth_to_save.save(depth_path)
            print(depth_path)

if __name__ == "__main__":
     tyro.cli(get_depth_for_dataset)