import argparse
import datetime
import os
import io
import random
import socket
import base64
import time
from glob import glob
import trimesh

# import hydra
import kaolin as kal
import ipdb  # noqa: F401
import numpy as np
import omegaconf
import torch
import torch.nn as nn
# from accelerate import Accelerator
from pytorch3d.renderer import PerspectiveCameras

from tools.co3d.utils.visualization import view_color_coded_images_from_tensor
from tools.co3d.utils.normalize import normalize_cameras_batch
from tools.co3d.utils.rays import cameras_to_rays
from tools.co3d.utils.visualization import (
    create_plotly_cameras_visualization,
    create_training_visualizations,
)

import torch.nn.functional as F
import open3d as o3d

import gzip
import json
import os.path as osp

import ipdb  # noqa: F401
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

import os
# Set the HF_ENDPOINT environment variable
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # before import all hf related pkgs
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


HTML_TEMPLATE = """
<html>
<head><meta charset="utf-8"/></head>
<body>
<img src="data:image/png;charset=utf-8;base64,{image_encoded}"/>
<img src="data:image/png;charset=utf-8;base64,{depth_encoded}"/>
<img src="data:image/png;charset=utf-8;base64,{proxy_encoded}"/>
{plotly_html}
</body>
</html>"""

CO3D_DATA_DIR = "/data1/yyb/raydiff/data"  # update this
CO3D_ANNOTATION_DIR = osp.join(CO3D_DATA_DIR, "co3d_v2_annotations")
CO3D_DIR = osp.join(CO3D_DATA_DIR, "CO3D_V2")
CO3D_ORDER_PATH = osp.join(
    CO3D_DATA_DIR, "co3d_v2_random_order_{sample_num}/{category}.json"
)

TRAINING_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]

TEST_CATEGORIES = [
    "ball",
    "book",
    "couch",
    "frisbee",
    "hotdog",
    "kite",
    "remote",
    "sandwich",
    "skateboard",
    "suitcase",
]

assert len(TRAINING_CATEGORIES) + len(TEST_CATEGORIES) == 51

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
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


#------------borrow from:
# https://github.com/facebookresearch/co3d/blob/b22f14538741cc806ff5e1783ab45f14c9ac9abe/dataset/co3d_dataset.py 

def _load_depth(path, scale_adjustment):
    if not path.lower().endswith(".jpg.geometric.png"):
        raise ValueError('unsupported depth file name "%s"' % path)

    d = _load_16big_png_depth(path) * scale_adjustment
    d[~np.isfinite(d)] = 0.0
    return d[None]  # fake feature channel


def _load_16big_png_depth(depth_png):
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth

def _load_1bit_png_mask(file: str):
    with Image.open(file) as pil_im:
        mask = (np.array(pil_im.convert("L")) > 0.0).astype(np.float32)
    return mask


def _load_depth_mask(path):
    if not path.lower().endswith(".png"):
        raise ValueError('unsupported depth mask file name "%s"' % path)
    m = _load_1bit_png_mask(path)
    return m[None]  # fake feature channel


#-----------------

def square_bbox(bbox, padding=0.0, astype=None):
    """
    Computes a square bounding box, with optional padding parameters.
    Args:
        bbox: Bounding box in xyxy format (4,).
    Returns:
        square_bbox in xyxy format (4,).
    """
    if astype is None:
        astype = type(bbox[0])
    bbox = np.array(bbox)
    center = (bbox[:2] + bbox[2:]) / 2
    extents = (bbox[2:] - bbox[:2]) / 2
    s = max(extents) * (1 + padding)
    square_bbox = np.array(
        [center[0] - s, center[1] - s, center[0] + s, center[1] + s],
        dtype=astype,
    )
    return square_bbox


def unnormalize_image(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    image = image * std + mean
    return (image * 255.0).astype(np.uint8)


def _transform_intrinsic(image, bbox, principal_point, focal_length):
    # Rescale intrinsics to match bbox
    half_box = np.array([image.width, image.height]).astype(np.float32) / 2
    org_scale = min(half_box).astype(np.float32)

    # Pixel coordinates
    principal_point_px = half_box - (np.array(principal_point) * org_scale)
    focal_length_px = np.array(focal_length) * org_scale
    principal_point_px -= bbox[:2]
    new_bbox = (bbox[2:] - bbox[:2]) / 2
    new_scale = min(new_bbox)

    # NDC coordinates
    new_principal_ndc = (new_bbox - principal_point_px) / new_scale
    new_focal_ndc = focal_length_px / new_scale

    principal_point = torch.tensor(new_principal_ndc.astype(np.float32))
    focal_length = torch.tensor(new_focal_ndc.astype(np.float32))

    return principal_point, focal_length


def construct_camera_from_batch(batch, device):
    if isinstance(device, int):
        device = f"cuda:{device}"

    return PerspectiveCameras(
        R=batch["R"].reshape(-1, 3, 3),
        T=batch["T"].reshape(-1, 3),
        focal_length=batch["focal_lengths"].reshape(-1, 2),
        principal_point=batch["principal_points"].reshape(-1, 2),
        image_size=batch["image_sizes"].reshape(-1, 2),
        device=device,
    )


def save_batch_images(images, fname):
    cmap = plt.get_cmap("hsv")
    num_frames = len(images)
    num_rows = len(images)
    num_cols = 4
    figsize = (num_cols * 2, num_rows * 2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()
    for i in range(num_rows):
        for j in range(4):
            if i < num_frames:
                axs[i * 4 + j].imshow(unnormalize_image(images[i][j]))
                for s in ["bottom", "top", "left", "right"]:
                    axs[i * 4 + j].spines[s].set_color(cmap(i / (num_frames)))
                    axs[i * 4 + j].spines[s].set_linewidth(5)
                axs[i * 4 + j].set_xticks([])
                axs[i * 4 + j].set_yticks([])
            else:
                axs[i * 4 + j].axis("off")
    plt.tight_layout()
    plt.savefig(fname)


def jitter_bbox(square_bbox, jitter_scale=(1.1, 1.2), jitter_trans=(-0.07, 0.07)):
    square_bbox = np.array(square_bbox.astype(float))
    s = np.random.uniform(jitter_scale[0], jitter_scale[1])
    tx, ty = np.random.uniform(jitter_trans[0], jitter_trans[1], size=2)
    side_length = square_bbox[2] - square_bbox[0]
    center = (square_bbox[:2] + square_bbox[2:]) / 2 + np.array([tx, ty]) * side_length
    extent = side_length / 2 * s
    ul = center - extent
    lr = ul + 2 * extent
    return np.concatenate((ul, lr))


class Co3dDataset(Dataset):
    def __init__(
        self,
        category=("all_train",),
        split="train",
        transform=None,
        num_images=2,
        img_size=224,
        mask_images=False,
        crop_images='bbox',
        co3d_dir=None,
        co3d_annotation_dir=None,
        apply_augmentation=True,
        normalize_cameras=True,
        no_images=False,
        sample_num=None,
        seed=0,
        load_extra_cameras=False,
    ):
        start_time = time.time()

        self.category = category
        self.split = split
        self.transform = transform
        self.num_images = num_images
        self.img_size = img_size
        self.mask_images = mask_images
        self.crop_images = crop_images
        self.apply_augmentation = apply_augmentation
        self.normalize_cameras = normalize_cameras
        self.no_images = no_images
        self.sample_num = sample_num
        self.load_extra_cameras = load_extra_cameras

        if self.apply_augmentation:
            self.jitter_scale = (1.1, 1.2)
            self.jitter_trans = (-0.07, 0.07)
        else:
            # Note if trained with apply_augmentation, we should still use
            # apply_augmentation at test time.
            self.jitter_scale = (1, 1)
            self.jitter_trans = (0.0, 0.0)

        if co3d_dir is not None:
            self.co3d_dir = co3d_dir
            self.co3d_annotation_dir = co3d_annotation_dir
        else:
            self.co3d_dir = CO3D_DIR
            self.co3d_annotation_dir = CO3D_ANNOTATION_DIR

        if isinstance(self.category, str):
            self.category = [self.category]

        if "all_train" in self.category:
            self.category = TRAINING_CATEGORIES
        if "all_test" in self.category:
            self.category = TEST_CATEGORIES
        if "full" in self.category:
            self.category = TRAINING_CATEGORIES + TEST_CATEGORIES
        self.category = sorted(self.category)
        self.is_single_category = len(self.category) == 1

        # Fixing seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.low_quality_translations = []
        self.rotations = {}
        self.category_map = {}
        for c in self.category:
            annotation_file = osp.join(
                self.co3d_annotation_dir, f"{c}_{self.split}.jgz"
            )
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())

            counter = 0
            for seq_name, seq_data in annotation.items():
                counter += 1
                if len(seq_data) < self.num_images:
                    continue

                filtered_data = []
                self.category_map[seq_name] = c
                bad_seq = False
                for data in seq_data:
                    # Make sure translations are not ridiculous and rotations are valid
                    det = np.linalg.det(data["R"])
                    if (np.abs(data["T"]) > 1e5).any() or det < 0.99 or det > 1.01:
                        bad_seq = True
                        self.low_quality_translations.append(seq_name)
                        break

                    # Ignore all unnecessary information.
                    filtered_data.append(
                        {
                            "filepath": data["filepath"],
                            "bbox": data["bbox"],
                            "R": data["R"],
                            "T": data["T"],
                            "focal_length": data["focal_length"],
                            "principal_point": data["principal_point"],
                        },
                    )

                if not bad_seq:
                    self.rotations[seq_name] = filtered_data

            print(f"Loaded {counter} instances of the {c} category.")

        self.sequence_list = list(self.rotations.keys())

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(self.img_size, antialias=True),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
            
            self.transform_d = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(self.img_size, antialias=True),
                ]
            )

        print(
            f"Low quality translation sequences, not used: {self.low_quality_translations}"
        )
        print(f"Data size: {len(self)}")
        print(f"Data loading took {(time.time()-start_time)} seconds.")

    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, index):
        num_to_load = self.num_images if not self.load_extra_cameras else 8

        sequence_name = self.sequence_list[index % len(self.sequence_list)]
        metadata = self.rotations[sequence_name]

        if self.sample_num is not None:
            with open(
                CO3D_ORDER_PATH.format(
                    sample_num=self.sample_num, category=self.category[0]
                )
            ) as f:
                order = json.load(f)
            ids = order[sequence_name][:num_to_load]
        else:
            ids = np.random.choice(len(metadata), num_to_load, replace=False)

        return self.get_data(index=index, ids=ids)

    def _get_scene_scale(self, sequence_name):
        n = len(self.rotations[sequence_name])

        R = torch.zeros(n, 3, 3)
        T = torch.zeros(n, 3)

        for i, ann in enumerate(self.rotations[sequence_name]):
            R[i, ...] = torch.tensor(self.rotations[sequence_name][i]["R"])
            T[i, ...] = torch.tensor(self.rotations[sequence_name][i]["T"])

        cameras = PerspectiveCameras(R=R, T=T)
        cc = cameras.get_camera_center()
        centeroid = torch.mean(cc, dim=0)
        diff = cc - centeroid

        norm = torch.norm(diff, dim=1)
        scale = torch.max(norm).item()

        return scale

    def _crop_image(self, image, bbox):
        image_crop = transforms.functional.crop(
            image,
            top=bbox[1],
            left=bbox[0],
            height=bbox[3] - bbox[1],
            width=bbox[2] - bbox[0],
        )
        return image_crop

    def _transform_intrinsic(self, image, bbox, principal_point, focal_length):
        half_box = np.array([image.width, image.height]).astype(np.float32) / 2
        org_scale = min(half_box).astype(np.float32)

        # Pixel coordinates
        principal_point_px = half_box - (np.array(principal_point) * org_scale)
        focal_length_px = np.array(focal_length) * org_scale
        principal_point_px -= bbox[:2]
        new_bbox = (bbox[2:] - bbox[:2]) / 2
        new_scale = min(new_bbox)

        # NDC coordinates
        new_principal_ndc = (new_bbox - principal_point_px) / new_scale
        new_focal_ndc = focal_length_px / new_scale

        return new_principal_ndc.astype(np.float32), new_focal_ndc.astype(np.float32)

    def get_data(self, index=None, sequence_name=None, ids=(0, 1), no_images=False):
        if sequence_name is None:
            index = index % len(self.sequence_list)
            sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]
        category = self.category_map[sequence_name]

        # Read image & camera information from annotations
        annos = [metadata[i] for i in ids]
        images = []
        depths = []
        image_sizes = []
        PP = []
        FL = []
        crop_parameters = []
        filenames = []
        for anno in annos:
            filepath = anno["filepath"]
            depthpath = filepath.replace('images', 'depths').replace('jpg', 'jpg.geometric.png')
            dmaskpath = filepath.replace('images', 'depth_masks').replace('jpg', 'png')
            if not no_images:
                image = Image.open(osp.join(self.co3d_dir, filepath)).convert("RGB")
                depth = Image.fromarray(_load_depth(osp.join(self.co3d_dir, depthpath), 1)[0])
                # Optionally mask images with black background
                if self.mask_images:
                    black_image = Image.new("RGB", image.size, (0, 0, 0))
                    mask_name = osp.basename(filepath.replace(".jpg", ".png"))

                    mask_path = osp.join(
                        self.co3d_dir, category, sequence_name, "masks", mask_name
                    )
                    mask = Image.open(mask_path).convert("L")

                    if mask.size != image.size:
                        mask = mask.resize(image.size)
                    mask = Image.fromarray(np.array(mask) > 125)
                    image = Image.composite(image, black_image, mask)

                # Determine crop, Resnet wants square images
                if  self.crop_images == 'bbox':
                    bbox_init = (anno["bbox"])
                elif self.crop_images == 'center':
                    
                    shorter_edge = min(image.width, image.height)
                    crop_size = shorter_edge
                    # Calculate the starting and ending coordinates for the crop
                    start_x = (image.width - crop_size) // 2
                    start_y = (image.height - crop_size) // 2
                    end_x = start_x + crop_size
                    end_y = start_y + crop_size
    
                    bbox_init = ([start_x, start_y, end_x, end_y])
                else:
                    bbox_init = ([0, 0, image.width, image.height])
        
                bbox = square_bbox(np.array(bbox_init)) #??
                if self.apply_augmentation:
                    bbox = jitter_bbox(
                        bbox,
                        jitter_scale=self.jitter_scale,
                        jitter_trans=self.jitter_trans,
                    )
                bbox = np.around(bbox).astype(int)

                # Crop parameters
                crop_center = (bbox[:2] + bbox[2:]) / 2
                # convert crop center to correspond to a "square" image
                width, height = image.size
                length = max(width, height)
                s = length / min(width, height) #
                crop_center = crop_center + (length - np.array([width, height])) / 2
                # convert to NDC
                cc = s * (1 - 2 * crop_center / length)
                crop_width = 2 * s * (bbox[2] - bbox[0]) / length
                crop_params = torch.tensor([-cc[0], -cc[1], crop_width, s]) # ??

                principal_point = torch.tensor(anno["principal_point"])
                focal_length = torch.tensor(anno["focal_length"])
                
                principal_point, focal_length = _transform_intrinsic(image, bbox, principal_point, focal_length)
                principal_point, focal_length = torch.tensor(principal_point), torch.tensor(focal_length)
                
                # Crop and normalize image
                image = self._crop_image(image, bbox)
                depth = self._crop_image(depth, bbox)
                # load and mask depth
                # mask =  Image.open(osp.join(self.co3d_dir, dmaskpath))
                # depth = Image.open(osp.join(self.co3d_dir, depthpath))
                image_224 = self.transform(image)
                depth_224 = self.transform_d(depth)
            
                # dmask = _load_depth_mask(osp.join(self.co3d_dir, dmaskpath))
                # depth = dmask * depth
                # plt.imsave('aaa.png', depth[0] / depth.max())
                rasize_scale = max(image.size) / max(image_224.shape)
                images.append(image_224[:, : self.img_size, : self.img_size])
                depths.append(depth_224[:, : self.img_size, : self.img_size])
                crop_parameters.append(crop_params)

            else:
                principal_point = torch.tensor(anno["principal_point"])
                focal_length = torch.tensor(anno["focal_length"])

            PP.append(principal_point)
            FL.append(focal_length)
            image_sizes.append(torch.tensor([self.img_size, self.img_size]))
            filenames.append(filepath)

        if self.load_extra_cameras:
            # Remove the extra loaded image, for saving space
            images = images[: self.num_images]
            depths = depths[: self.num_images]

        images = torch.stack(images)
        depths = torch.stack(depths)
        crop_parameters = torch.stack(crop_parameters)

            
        pcd_path = osp.join(
            self.co3d_dir, category, sequence_name, "pointcloud.ply"
            )

        # Assemble batch info to send back
        R = torch.stack([torch.tensor(anno["R"]) for anno in annos])
        T = torch.stack([torch.tensor(anno["T"]) for anno in annos])
        focal_lengths = torch.stack(FL)
        principal_points = torch.stack(PP)
        image_sizes = torch.stack(image_sizes)

        batch = {
            "model_id": sequence_name,
            "category": category,
            "n": len(metadata),
            "ind": torch.tensor(ids),
            "image": images,
            "depth": depths,
            "R": R,
            "T": T,
            "focal_length": focal_lengths,
            "principal_point": principal_points,
            "image_size": image_sizes,
            "crop_parameters": crop_parameters,
            "filename": filenames,
            "pcd_path":pcd_path
        }

        return batch


class MeanPoolingDownsample(nn.Module):
    def __init__(self, downsample_factor):
        super(MeanPoolingDownsample, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor)

    def forward(self, x):
        return self.pool(x)


def gaussian_kernel(size, sigma):
    x = np.linspace(-(size // 2), size // 2, size)
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_2d /= kernel_2d.sum()
    return kernel_2d

class GaussianBlur(nn.Module):
    def __init__(self, kernel_size, sigma):
        super(GaussianBlur, self).__init__()
        kernel = gaussian_kernel(kernel_size, sigma)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, self.weight, padding=self.weight.shape[2]//2)
        return x
    

def fill_invalid_depth(depth_map ,mask):
    """
    将深度图中无效的深度值（depth == 0）用最近的有效像素填充。
    Args:
        depth_map (torch.Tensor): 形状为 (b, 1, h, w) 的深度图。

    Returns:
        torch.Tensor: 填充后的深度图。
    """
    b, c, h, w = depth_map.shape
    assert c == 1, "深度图的通道数应为1"

    # 创建一个二值掩码，标识有效的深度值
    valid_mask = mask
    invalid_mask =~mask

    # 创建一个坐标网格
    y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w))
    y_grid = y_grid.to(depth_map.device)
    x_grid = x_grid.to(depth_map.device)

    # 初始化填充后的深度图
    filled_depth_map = depth_map.clone()

    for i in range(b):
        # 获取有效深度值和对应的坐标
        valid_depth = depth_map[i, 0, valid_mask[i, 0]]
        valid_y = y_grid[valid_mask[i, 0]]
        valid_x = x_grid[valid_mask[i, 0]]

        # 将有效深度值和对应的坐标拼接成 (N, 2) 的张量
        valid_points = torch.stack([valid_y, valid_x], dim=-1).float()

        # 将网格坐标拼接成 (h, w, 2) 的张量
        grid_points = torch.stack([y_grid, x_grid], dim=-1).float().view(-1, 2)

        # 计算最近邻插值
        distances = torch.cdist(grid_points, valid_points)
        nearest_indices = torch.argmin(distances, dim=-1)
        nearest_depths = valid_depth[nearest_indices]

        # 填充无效的深度值
        filled_depth_map[i, 0, invalid_mask[i, 0]] = nearest_depths[invalid_mask[i, 0].view(-1)]

    return filled_depth_map




def plotly_scene_visualization(R_pred, T_pred, pcd = None, mesh = None):
    num_frames = len(R_pred)

    camera = {}
    for i in range(num_frames):
        camera[i] = PerspectiveCameras(R=R_pred[i, None], T=T_pred[i, None])

    geo_to_plot =  {**camera}
    if pcd is not None:
        geo_to_plot.update({"pcd":pcd})
    if mesh is not None:
        geo_to_plot.update({"mesh":mesh})
        
    fig = plot_scene(
       {"scene" : geo_to_plot},
        camera_scale=0.1,
    )
    fig.update_scenes(aspectmode="cube")

    cmap = plt.get_cmap("hsv")
    for i in range(num_frames):
        fig.data[i].line.color = matplotlib.colors.to_hex(cmap(i / (num_frames)))
    
        
    return fig


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="examples/robot/images")
    parser.add_argument("--model_dir", type=str, default="models/co3d_diffusion")
    parser.add_argument("--mask_dir", type=str, default="")
    parser.add_argument("--bbox_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="output_cameras.html")
    return parser


def main(image_dir, model_dir, mask_dir, bbox_path, output_path):
    
    num_images = 8
    random_num_images = False
    append_ndc = True
    load_extra_cameras = False
    translation_scale = 1
    normalize_first_camera = False
    num_patches_x, num_patches_y = 16, 16
    device = 'cuda:0'
        
    #image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device)
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device)
    gaussian_blur = GaussianBlur(kernel_size=5, sigma=1.0).to(device)
    
    dataset = Co3dDataset(
            # category="hydrant",
            category="apple",
            split="train",
            num_images=num_images,
            apply_augmentation=False,
            crop_images='center',
            load_extra_cameras=load_extra_cameras,
        )
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

    #TODO Load Zoe and SAM here?
    print('Load dataset done')
    for batch in dataloader:
        images = batch["image"].to(device)
        depths = batch["depth"].to(device)
        focal_lengths = batch["focal_length"].to(device)
        crop_params = batch["crop_parameters"].to(device)
        principal_points = batch["principal_point"].to(device)
        R = batch["R"].to(device)
        T = batch["T"].to(device)
        batch_size = batch["T"].shape[0]
        pcd_path = batch["pcd_path"]

            
        depths, images = depths[0], images[0]
        
        if True:
            # for gt depth
            _min, _max = torch.quantile(
                depths[depths > 0],
                torch.tensor([0.02, 0.98]).to(device),
            )
            _min, _max = _min.item(), _max.item()
        # # save proxycreate_oriented_bounding_box_o3d
        # depths_coarse = fill_invalid_depth(depths, mask = depths > 0.1 ) 

            
        # prepare image for the model
        # inputs = image_processor(images=images, return_tensors="pt")
        with torch.no_grad():
            inputs = dict(pixel_values = images)
            outputs = model(**inputs)
        pred_depth = outputs.predicted_depth
        max_ = pred_depth.reshape(pred_depth.shape[0], -1).max(dim = -1)[0]
        pred_depth = pred_depth[:,None] / max_[:,None,None,None]
        
        depth_proxy = F.interpolate(pred_depth, (16,16), mode = 'bilinear', antialias=True)

        if True:
            s = io.BytesIO()
            view_color_coded_images_from_tensor(images)
            plt.savefig(s, format="png", bbox_inches="tight")
            plt.close()
            image_encoded = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
            
            s = io.BytesIO()
            depths_proxy_colored = colorize_depth_maps(depth_proxy, 0, 1)
            view_color_coded_images_from_tensor(depths_proxy_colored)
            plt.savefig(s, format="png", bbox_inches="tight")
            plt.close()
            proxy_encoded = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
            
            # save depth
            s = io.BytesIO()
            depths_colored = colorize_depth_maps(depths,  min_depth = _min , max_depth= _max)
            view_color_coded_images_from_tensor(depths_colored)
            plt.savefig(s, format="png", bbox_inches="tight")
            plt.close()
            depth_encoded = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
            
            with open(output_path, "w") as f:
                s = HTML_TEMPLATE.format(
                    image_encoded=image_encoded,
                    depth_encoded =depth_encoded,
                    proxy_encoded =proxy_encoded,
                    plotly_html='',
                )
                f.write(s)
        print('?')
            
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))


