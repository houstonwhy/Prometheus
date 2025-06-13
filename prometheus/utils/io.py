#TODO 
"""Contains utility functions for  load & export
"""
import os
from io import BytesIO
import importlib
import imageio
import numpy as np
import torch
import tqdm
import einops
from plyfile import PlyData, PlyElement

from .math import inverse_sigmoid
from .camera import sample_from_dense_cameras
from .image_utils import colorize_depth_maps

def import_str(string):
    # From https://github.com/CompVis/taming-transformers
    module, cls = string.rsplit(".", 1)
    if 'plaonicgen.' in module:
        module = module.replace("plaonicgen.", "prometheus.")
    return getattr(importlib.import_module(module, package=None), cls)

def export_ply_for_gaussians(path, gaussians, opacity_threshold=0.00, bbox=[-2, 2], compatible=True):

    xyz, features, opacity, scales, rotations = gaussians

    # assert xyz.shape[0] == 1
     
    means3D = xyz[0].contiguous().float()
    opacity = opacity[0].contiguous().float()
    scales = scales[0].contiguous().float()
    rotations = rotations[0].contiguous().float()
    shs = features[0].contiguous().float() # [N, 1, 3]

    # prune by opacity
    mask = opacity[..., 0] >= opacity_threshold
    means3D = means3D[mask]
    opacity = opacity[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    shs = shs[mask]

    # invert activation to make it compatible with the original ply format
    if compatible:
        opacity = inverse_sigmoid(opacity)
        scales = torch.log(scales + 1e-8)

    xyzs = means3D.detach().cpu().numpy()
    f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    rotations = rotations.detach().cpu().numpy()

    l = ['x', 'y', 'z']
    # All channels except the 3 DC
    for i in range(f_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    l.append('opacity')
    for i in range(scales.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotations.shape[1]):
        l.append('rot_{}'.format(i))

    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(xyzs.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')

    # PlyData([el]).write(path + '.ply')

    plydata = PlyData([el])

    vert = plydata["vertex"]
    sorted_indices = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
        / (1 + np.exp(-vert["opacity"]))
    )
    buffer = BytesIO()
    for idx in sorted_indices:
        v = plydata["vertex"][idx]
        position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
        scales = np.exp(
            np.array(
                [v["scale_0"], v["scale_1"], v["scale_2"]],
                dtype=np.float32,
            )
        )
        rot = np.array(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            dtype=np.float32,
        )
        SH_C0 = 0.28209479177387814
        color = np.array(
            [
                v["f_dc_0"],
                v["f_dc_1"],
                v["f_dc_2"],
                # 0.5 + SH_C0 * v["f_dc_0"],
                # 0.5 + SH_C0 * v["f_dc_1"],
                # 0.5 + SH_C0 * v["f_dc_2"],
                1 / (1 + np.exp(-v["opacity"])),
            ]
        )
        buffer.write(position.tobytes())
        buffer.write(scales.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )

    with open(path + '.splat', "wb") as f:
        f.write(buffer.getvalue())
    
    return {
        'pts' : attributes[:,0:3],
        'color' :(attributes[:,3:6] + 1) / 2,
        'alpha' : 1 / (1 + np.exp(-attributes[:,6:7]))
    }



def load_ply_for_gaussians(path, device='cpu', compatible=True):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    print("Number of points at loading : ", xyz.shape[0])

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = torch.tensor(xyz, dtype=torch.float, device=device)[None]
    features = torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2)[None]
    opacity = torch.tensor(opacities, dtype=torch.float, device=device)[None]
    scales = torch.tensor(scales, dtype=torch.float, device=device)[None]
    rotations = torch.tensor(rots, dtype=torch.float, device=device)[None]

    if compatible:
        opacity = torch.sigmoid(opacity)
        scales = torch.exp(scales)

    return xyz, features, opacity, scales, rotations


@torch.no_grad()
def export_video(render_fn, save_path, name, dense_cameras, fps=60, num_frames=720, render_size=512, device='cuda:0'):

    images = []
    depths = []
    dense_cameras = dense_cameras.to(device)
    for i in tqdm.trange(num_frames, desc="Rendering video..."):

        t = torch.full((1, 1), fill_value=i/num_frames, device=device)

        camera = sample_from_dense_cameras(dense_cameras, t)
        
        image, depth = render_fn(camera, render_size, render_size)

        images.append(image.reshape(3, render_size, render_size).permute(1,2,0).detach().cpu().mul(1/2).add(1/2).clamp(0, 1).mul(255).numpy().astype(np.uint8))
        depths.append(depth.reshape(1, render_size, render_size).permute(1,2,0).detach().cpu().numpy())
    depths = np.stack(depths)
    depth_vis = (colorize_depth_maps(depths, max_depth=depths.max(), min_depth=depths.min()+1e-3).clip(0,1) * 255).astype(np.uint8)
    depth_vis = einops.rearrange(depth_vis, 'B C H W -> B H W C')
    #depth_vis = [depth_vis[i] for i in range(depth_vis.shape[0])]
    imageio.mimwrite(os.path.join(save_path, f'{name}.mp4'), images, fps=fps, quality=8, macro_block_size=1)
    imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'),depth_vis, fps=fps, quality=5, macro_block_size=1)

    return images, depths

