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

import open3d as o3d
from pytorch3d.transforms import Transform3d
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.structures import (
    Pointclouds,
    # TexturesVertex,
    Meshes
    )
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    # DepthShader,
    PointsRasterizer,
    AlphaCompositor,
    TexturesVertex
)

from pytorch3d.renderer import (
    # look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)

from sklearn.decomposition import PCA

HTML_TEMPLATE = """
<html>
<head><meta charset="utf-8"/></head>
<body>
<img src="data:image/png;charset=utf-8;base64,{image_encoded}"/>
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


def remove_outliers(points, z_thresh=2.0):
    """
    使用 Z-score 方法去除离群点。
    """
    mean = torch.mean(points, dim=0)
    std = torch.std(points, dim=0)
    z_scores = torch.abs((points - mean) / std)
    inliers = z_scores < z_thresh
    inliers = torch.all(inliers, dim=1)
    return points[inliers]


def merge_meshes(meshes_batch):
    """
    将批次大小为 2 的 Meshes 对象合并成一个单一的 Meshes 对象。

    参数:
    meshes_batch (Meshes): 批次大小为 2 的 Meshes 对象。

    返回:
    Meshes: 合并后的 Meshes 对象。
    """
    verts_list = []
    faces_list = []
    textures_features_list = []
    verts_num = 0
    for mesh in meshes_batch:
        verts_list.append(mesh.verts_list()[0])
        faces_list.append(mesh.faces_list()[0] + verts_num)
        textures_features_list.append(mesh.textures.verts_features_list()[0])
        verts_num += mesh.verts_list()[0].shape[0]

    merged_verts = torch.cat(verts_list, dim=0)
    merged_faces = torch.cat(faces_list, dim=0)
    merged_textures_features = torch.cat(textures_features_list, dim=0)

    merged_textures = TexturesVertex(verts_features=merged_textures_features[None])
    merged_mesh = Meshes(verts=[merged_verts], faces=[merged_faces], textures=merged_textures)
    return merged_mesh


def render_meshes(meshes, cameras, image_size=512, device = 'cuda'):
    """
    使用多个相机从不同位置渲染场景中的多个网格。

    参数:
    meshes (Meshes): PyTorch3D 的 Meshes 对象。
    camera_positions (list of tuples): 相机位置列表，每个元素是一个 (R, T) 元组。
    image_size (int): 渲染图像的大小。

    返回:
    list of torch.Tensor: 渲染图像列表。
    """
    # 设置渲染器
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    rendered_images, rendered_depths = [], []
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )
    mesh = merge_meshes(meshes)
    for i in range(len(cameras)):
        # 为每个相机创建一个渲染器
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras[i], raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=cameras[i], lights=lights)
        )
        image = renderer(mesh)
        rendered_images.append(image)
        # 渲染深度图
        rasterizer = MeshRasterizer(
            cameras=cameras[i], raster_settings=raster_settings
        )
        fragments = rasterizer(mesh)
        depth = fragments.zbuf
        rendered_depths.append(depth)

    rendered_images = torch.cat(rendered_images, dim=0)
    rendered_depths = torch.cat(rendered_depths, dim=0)
    
    depths_mask = rendered_depths != -1
    d_max, d_min = rendered_depths[depths_mask].max(), rendered_depths[depths_mask].min()
    rendered_depths[depths_mask] = (rendered_depths[depths_mask]  - d_min) / (d_max - d_min)
    return rendered_images, rendered_depths

def open3d_to_pytorch3d(mesh_o3d_list):
    """
    将Open3D的网格列表转换为PyTorch3D的格式。

    参数:
    mesh_o3d_list (list of open3d.geometry.TriangleMesh): Open3D的网格对象列表。

    返回:
    pytorch3d.structures.Meshes: PyTorch3D的网格对象。
    """
    verts_list = []
    faces_list = []
    textures_list = []

    for mesh_o3d in mesh_o3d_list:
        # 获取顶点和面并转换为PyTorch张量
        verts_o3d = np.asarray(mesh_o3d.vertices)
        faces_o3d = np.asarray(mesh_o3d.triangles)

        verts_torch = torch.tensor(verts_o3d, dtype=torch.float32)
        faces_torch = torch.tensor(faces_o3d, dtype=torch.int64)

        # 创建纹理（这里简单地使用顶点颜色）
        if mesh_o3d.has_vertex_colors():
            colors_o3d = np.asarray(mesh_o3d.vertex_colors)
            colors_torch = torch.tensor(colors_o3d, dtype=torch.float32)
        else:
            colors_torch = torch.ones_like(verts_torch)  # 
        #textures = TexturesVertex(verts_features=colors_torch.unsqueeze(0))

        verts_list.append(verts_torch)
        faces_list.append(faces_torch)
        textures_list.append(colors_torch)

    textures_list = torch.stack(textures_list, dim=0)
    textures = TexturesVertex(verts_features=textures_list)
    # 创建PyTorch3D的Mesh对象
    mesh_torch3d = Meshes(verts=verts_list, faces=faces_list, textures=textures)

    return mesh_torch3d


def create_mesh_from_bbox(obb, color=(1, 0, 0)):
    """
    将 OrientedBoundingBox 转换为 LineSet 并设置颜色。

    参数:
    obb (open3d.geometry.OrientedBoundingBox): 定向边界框对象。
    color (tuple): 颜色 (R, G, B)，默认为红色 (1, 0, 0)。

    返回:
    open3d.geometry.LineSet: 表示定向边界框的 LineSet 对象。
    """

    #bbox_points = np.asarray(obb.get_box_points())
    # bbox_lines = [
    #     [0, 1], [1, 2], [2, 3], [3, 0],
    #     [4, 5], [5, 6], [6, 7], [7, 4],
    #     [0, 4], [1, 5], [2, 6], [3, 7]
    # ]
    # bbox_colors = [color for _ in range(len(bbox_lines))]
    bbox_points = np.array(
      [[-0.5, -0.5, -0.5],
       [ 0.5, -0.5, -0.5],
       [ 0.5,  0.5, -0.5],
       [-0.5,  0.5, -0.5],
       [-0.5, -0.5,  0.5],
       [ 0.5, -0.5,  0.5],
       [ 0.5,  0.5,  0.5],
       [-0.5,  0.5,  0.5]]) # 
        #      7--------6
        #     /|       /|
        #    / |      / |
        #   3--------2  |
        #   |  4-----|--5
        #   | /      | /
        #   |/       |/
        #   0--------1
    #bbox_points = np.asarray(object_box.get_box_points())-> o3d return worng oredr with default bbox points setting
    try:
        bbox_points =(obb.extent * bbox_points) @ obb.R.T  + obb.center
    except:
        bbox_points = np.asarray(obb.get_box_points())
    
    #bbox_points = np.asarray(obb.get_box_points())
    bbox_faces = np.array( [
    [0, 1, 2], [0, 2, 3],  # 前面 (z_min)
    [4, 5, 6], [4, 6, 7],  # 后面 (z_max)
    [0, 1, 5], [0, 5, 4],  # 左面 (x_min)
    [2, 3, 7], [2, 7, 6],  # 右面 (x_max)
    [0, 3, 7], [0, 7, 4],  # 上面 (y_max)
    [1, 2, 6], [1, 6, 5]   # 下面 (y_min)
    ])

    bbox_mesh = o3d.geometry.TriangleMesh()
    bbox_mesh.vertices = o3d.utility.Vector3dVector(bbox_points)
    bbox_mesh.triangles = o3d.utility.Vector3iVector(bbox_faces)

    bbox_mesh.compute_vertex_normals()
    bbox_mesh.paint_uniform_color(color)

    return bbox_mesh

def create_oriented_bounding_box_o3d(pcd, cameras = None, c = True, with_ground = False):
    points = pcd.points_padded().squeeze(0).cpu()
    #points = remove_outliers(points).numpy()
    
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(points)
    object_box = pcd_o3d.get_oriented_bounding_box() 
    # bbox_points = np.array(
    #   [[-0.5, -0.5, -0.5],
    #    [-0.5, -0.5,  0.5],
    #    [-0.5,  0.5, -0.5],
    #    [-0.5,  0.5,  0.5],
    #    [ 0.5, -0.5, -0.5],
    #    [ 0.5, -0.5,  0.5],
    #    [ 0.5,  0.5, -0.5],
    #    [ 0.5,  0.5,  0.5]])
    # #bbox_points = np.asarray(object_box.get_box_points())-> o3d return worng oredr with default bbox points setting
    # bbox_points =(object_box.extent * bbox_points) @ object_box.R.T  + object_box.center
    
    object_mesh = create_mesh_from_bbox(object_box, [1, 0, 0])

    #------------------------fit the ground bbox based on bbox and cameras look-at vector-------
    if with_ground  and cameras is not None:        
        # find the look-at plane
        c2w = cameras.get_world_to_view_transform().inverse().get_matrix().cpu().numpy()
        cameras_points = c2w[:,3,:3]
        cameras_lookat = c2w[:,0,:3] # look-at in pyotch3d is z+
        # init normal vec table
        resolution = 6
        x = torch.linspace(-1, 1, resolution)
        y = torch.linspace(-1, 1, resolution)
        z = torch.linspace(-1, 1, resolution)

        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        normal_vectors = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)
        normal_vectors = normal_vectors / torch.norm(normal_vectors, dim=1, keepdim=True)

        # find best normal vec 
        dot_products = normal_vectors @ cameras_lookat.T
        # sum_dot_products = torch.sum(dot_products, dim=1)
        sum_dot_products  = dot_products.norm(2, dim = 1)
        # find the normal vector of the ground plane(all camers look-at)
        best_normal_vector_index = torch.argmin(sum_dot_products)
        ground_vec = normal_vectors[best_normal_vector_index] # we assume this vec is the normal vector of ground
        ground2world = create_new_coordinate_system(z = -ground_vec) # c2 denote the "ground cooriante system"
        world2ground = np.linalg.inv(ground2world)
        #
        ground_thickness = 0.1  
        bbox_points = np.asarray(object_box.get_box_points())
        full_points = np.concatenate([bbox_points, cameras_points], axis = 0)
        full_points = full_points @ world2ground[:3,:3]
        min_bound, max_bound = full_points.min(axis=0),  full_points.max(axis=0)
        max_bound[2] = min_bound[2] +   ground_thickness
    
        ground_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        ground_mesh = create_mesh_from_bbox(ground_bbox, [0, 0, 0])
        ground_mesh.rotate(ground2world[:3,:3], center=ground_mesh.get_center())

        full_mesh = open3d_to_pytorch3d([object_mesh, ground_mesh])
        full_mesh = merge_meshes(full_mesh)
    #---------------------------------------------------------------------------
    else:
        full_mesh = open3d_to_pytorch3d([object_mesh])
        
    return full_mesh

def create_new_coordinate_system(z):
    # Step 1: Normalize the z vector
    z_norm = z / torch.norm(z)

    # Step 2: Choose a reference vector
    ref = torch.tensor([1.0, 0.0, 0.0])

    # Step 3: Calculate the x vector
    x = ref - torch.dot(ref, z_norm) * z_norm
    x_norm = x / torch.norm(x)

    # Step 4: Calculate the y vector
    y = torch.cross(z_norm, x_norm)
    y_norm = y / torch.norm(y)

    R = torch.stack([x_norm, y_norm, z_norm])
    
    RT = torch.eye(4)
    RT[:3,:3] = R
    return RT


def create_oriented_bounding_box(pcd, cameras = None, color = (1,0,0), return_mesh = True):
    pcd = pcd.cpu()
    # 获取点云的点
    points = pcd.points_padded().squeeze(0)
    points = remove_outliers(points)
    # 使用 PCA 计算点云的主轴
    pca = PCA(n_components=3)
    pca.fit(points)

    # 获取主轴方向
    principal_axes = torch.tensor(pca.components_, dtype=torch.float32)

    # 计算点云的中心
    center = points.mean(dim=0)

    # 计算点云在每个主轴方向上的最小和最大投影
    projections = points @ principal_axes.T
    min_projections = projections.min(dim=0).values
    max_projections = projections.max(dim=0).values

    # 计算每个主轴方向上的投影范围
    projection_ranges = max_projections - min_projections

    # 计算投影范围的最大值
    # max_range = projection_ranges.max()
    max_range = max_projections - min_projections

    # 计算每个主轴方向上的中心投影
    center_projections = (min_projections + max_projections) / 2

    # 创建旋转包围盒的顶点
    obb_vertices = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                vertex = center 
                + principal_axes[0] * (center_projections[0] + (max_range[0] / 2 if i == 1 else -max_range[0] / 2)) \
                + principal_axes[1] * (center_projections[1] + (max_range[1] / 2 if j == 1 else -max_range[1] / 2)) \
                + principal_axes[2] * (center_projections[2] + (max_range[2] / 2 if k == 1 else -max_range[2] / 2))
                obb_vertices.append(vertex)

    obb_vertices = torch.stack(obb_vertices)
    
    
    
    def create_obb_mesh(obb_vertices, color):
        # 定义包围盒的面（每个面由两个三角形组成）
        faces = torch.tensor([
            [0, 1, 2], [1, 3, 2],  # 前
            [4, 6, 5], [5, 6, 7],  # 后
            [0, 2, 4], [2, 6, 4],  # 左
            [1, 5, 3], [3, 5, 7],  # 右
            [0, 4, 1], [1, 4, 5],  # 下
            [2, 3, 6], [3, 7, 6]   # 上
        ], dtype=torch.int64)

        # 创建网
        # 手动指定顶点颜色
        vertex_colors = torch.tensor([color for i in range(8)], dtype=torch.float32)
        textures = TexturesVertex(verts_features=[vertex_colors])
        
        mesh = Meshes(verts=[obb_vertices], faces=[faces], textures=textures)
        return mesh

    
    if return_mesh:
        return create_obb_mesh(obb_vertices, color)
    else:
        
        return obb_vertices
    

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
        image_sizes = []
        PP = []
        FL = []
        crop_parameters = []
        filenames = []
        for anno in annos:
            filepath = anno["filepath"]

            if not no_images:
                image = Image.open(osp.join(self.co3d_dir, filepath)).convert("RGB")

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
                image_224 = self.transform(image)
                rasize_scale = max(image.size) / max(image_224.shape)
                images.append(image_224[:, : self.img_size, : self.img_size])
                crop_parameters.append(crop_params)

            else:
                principal_point = torch.tensor(anno["principal_point"])
                focal_length = torch.tensor(anno["focal_length"])

            PP.append(principal_point)
            FL.append(focal_length)
            image_sizes.append(torch.tensor([self.img_size, self.img_size]))
            filenames.append(filepath)

        if not no_images:
            if self.load_extra_cameras:
                # Remove the extra loaded image, for saving space
                images = images[: self.num_images]

            images = torch.stack(images)
            crop_parameters = torch.stack(crop_parameters)
        else:
            images = None
            crop_parameters = None
            
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
    
    num_images = 100
    random_num_images = False
    append_ndc = True
    load_extra_cameras = False
    translation_scale = 1
    normalize_first_camera = False
    num_patches_x, num_patches_y = 16, 16
    
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
    device = 'cuda:0'
    for batch in dataloader:
        images = batch["image"].to(device)
        focal_lengths = batch["focal_length"].to(device)
        crop_params = batch["crop_parameters"].to(device)
        principal_points = batch["principal_point"].to(device)
        R = batch["R"].to(device)
        T = batch["T"].to(device)
        batch_size = batch["T"].shape[0]
        pcd_path = batch["pcd_path"]
        cameras_og = [
            PerspectiveCameras(
                focal_length=focal_lengths[b],
                principal_point=principal_points[b],
                R=R[b],
                T=T[b],
                device=device,
            )
            for b in range(batch_size)
        ]


        cameras, undo_transform = normalize_cameras_batch(
                cameras=cameras_og,
                scale=translation_scale,
                normalize_first_camera=normalize_first_camera,
            )
            
        cameras, images = cameras[0], images[0]
        if num_images > 1:
            a, normalize_tr, normalize_s = undo_transform[0](cameras)
            normalize_tr._matrix = normalize_tr._matrix.inverse()
            #normalize_tr = normalize_tr.scale(1/normalize_s).to(device)
        else:
            normalize_s = 1
            normalize_tr = Transform3d(device=device)
            normalize_tr._matrix = torch.eye(4, device=device)[None,:,:]

        # load pcd -> apply normalization -> compute proxy
        mesh = trimesh.load(pcd_path[0])
        points = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)
        colors = torch.tensor(mesh.visual.vertex_colors[:, :3] / 255.0, dtype=torch.float32).to(device)
        # pcd_og = Pointclouds(points=[points], features=[colors])
        points = normalize_tr.transform_points(points) / normalize_s
        
        colors = torch.tensor(mesh.visual.vertex_colors[:, :3] / 255.0, dtype=torch.float32).to(device)
        pcd = Pointclouds(points=[points], 
                          features=[colors])
        meshes = create_oriented_bounding_box_o3d(pcd, cameras = cameras).to(device)
        #meshes = create_oriented_bounding_box(pcd, cameras = cameras).to(device)
        
        if True:
            proxy_rgb, proxy_depth = render_meshes(meshes=meshes, cameras=cameras, image_size=images.shape[-1], device = device)
        else:
            rasterizer = PointsRasterizer(cameras=cameras, raster_settings=None)
            renderer = PointsRenderer(
                rasterizer=rasterizer,
                compositor=AlphaCompositor()
            )
            # 
            pcds_ = Pointclouds(points=[points] * images.shape[0], 
                          features=[colors] * images.shape[0])
            proxy_rgb = renderer(pcds_)
                
        if True:
            fig = plotly_scene_visualization(cameras.R, 
                                             cameras.T, 
                                             pcd = pcd, 
                                             mesh = meshes)
            html_plot = plotly.io.to_html(fig, full_html=False, include_plotlyjs="cdn")
            # save rgb
            s = io.BytesIO()
            view_color_coded_images_from_tensor(images)
            plt.savefig(s, format="png", bbox_inches="tight")
            plt.close()
            image_encoded = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
            
            # save proxycreate_oriented_bounding_box_o3d
            s = io.BytesIO()
            view_color_coded_images_from_tensor(proxy_depth[...,:])
            plt.savefig(s, format="png", bbox_inches="tight")
            plt.close()
            proxy_encoded = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
            
            with open(output_path, "w") as f:
                s = HTML_TEMPLATE.format(
                    image_encoded=image_encoded,
                    proxy_encoded =proxy_encoded,
                    plotly_html=html_plot,
                )
                f.write(s)
        print('?')
            
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))


