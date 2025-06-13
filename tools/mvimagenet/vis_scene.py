import os
import json
import tyro
import numpy as np
from tqdm.auto import tqdm
from typing import NamedTuple
import ipdb  # noqa: F401
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go

import torch
# import open3d as o3d
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
    PointsRasterizer,
    AlphaCompositor,
    TexturesVertex
)

from tools.mvimagenet.poses.pose_utils import gen_poses, load_colmap_data

def llff2opengl(poses):
    """
    Convert camera poses from LLFF coordinate system to OpenGL coordinate system.
    
    Returns:
    numpy.ndarray: A 3x4 or Nx3x4 array of camera poses in OpenGL coordinate system.
    """
    # Swap the Y and Z axes and negate the Z axis
    if isinstance(poses, torch.Tensor):
        return torch.cat([poses[:, : ,1:2], -poses[:, :, 0:1], poses[:, :, 2:3], poses[:, :, 3:]], dim=2)
    return np.concatenate([poses[:, : ,1:2], -poses[:, :, 0:1], poses[:, :, 2:3], poses[:, :, 3:]], axis=2)

def matrix_to_square(mat):
    l = len(mat.shape)
    if l==3:
        return torch.cat([mat, torch.tensor([0,0,0,1]).repeat(mat.shape[0],1,1).to(mat.device)],dim=1)
    elif l==4:
        return torch.cat([mat, torch.tensor([0,0,0,1]).repeat(mat.shape[0],mat.shape[1],1,1).to(mat.device)],dim=2)

class AxisArgs(NamedTuple):  # pragma: no cover
    showgrid: bool = False
    zeroline: bool = False
    showline: bool = False
    ticks: str = ""
    showticklabels: bool = False
    backgroundcolor: str = "#fff"
    showaxeslabels: bool = False


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
        
    array_axs =[1, 0.5, 0,-0.5, -1] 
    array_axs = [x * 1.5 for x in array_axs]
    range_axs = [-1, 1]
    range_axs = [x * 1.5 for x in range_axs]
    fig = plot_scene(
       {"scene" : geo_to_plot},
        camera_scale=0.1,
                yaxis={ "title": "Y",
                "backgroundcolor":"rgb(200, 200, 230)",    
                'tickmode': 'array',
                'tickvals': array_axs,
                'range':range_axs,
        },
        zaxis={ "title": "Z",
                'tickmode': 'array',
                'tickvals': array_axs,
                'range':range_axs,
        },
        xaxis={ "title": "X",
                'tickmode': 'array',
                'tickvals': array_axs,
                'range':range_axs,
        },
        axis_args=AxisArgs(showline=False,showgrid=True,zeroline=True,showticklabels=False,showaxeslabels=True),
    )
    fig.update_scenes(aspectmode="data")

    cmap = plt.get_cmap("Spectral")
    for i in range(num_frames):
        #color = cmap(int(cmap.N * (i / (num_frames - 1)) % cmap.N))
        color = cmap(i / (num_frames - 1))
        #fig.data[i].line.color = color
        fig.data[i].line.color = matplotlib.colors.to_hex(color)

    return fig

def get_w2c(camera):
    w2c = np.linalg.inv(camera)
    return w2c

if __name__ == "__main__":
    dataset_root = "/data1/yyb/datasets/MVImgNet"
    category = '99'
    scene_id = '1001290e'
    device = 'cuda:0'
    normalized_cameras = True
    
    if category == 'all':
        categories = os.listdir(dataset_root)
    else:
        categories = [category]
    basedirs = []
    
    if scene_id != '':
        basedirs.append(os.path.join(dataset_root, category, scene_id))
    else:
        basedirs = []
        for category in categories:
                if not os.path.isdir(os.path.join(dataset_root, category)):
                    continue
                for id in os.listdir(os.path.join(dataset_root, category)):
                    if not os.path.isdir(os.path.join(dataset_root, category, id)):
                        continue
                    basedirs.append(os.path.join(dataset_root, category, id))
    
    for scene in tqdm(basedirs):
        print(scene)
        if not os.path.exists(os.path.join(scene, 'sparse/0')):
            continue
        files_had = os.listdir(os.path.join(scene, 'sparse/0'))
        poses, pts3d, perm = load_colmap_data(scene, load_pcd=True)
        # load pcd
        points = torch.tensor([p.xyz for p in pts3d.values()]).to(device)
        colors = torch.tensor([p.rgb for p in pts3d.values()]).to(device) / 255
        # load cameras
        if False: 
            c2ws, hwfs = poses.transpose((2,0,1))[perm,:3,:4], poses.transpose((2,0,1))[perm,:3,4]
        else:
            with open(os.path.join(scene, 'cameras.json'), 'r') as fp:
                cameras= json.load(fp)
            perm = np.array(([int(k) for k in cameras.keys()]))
            c2ws = np.array([v['c2w'] for k, v in cameras.items()])[perm]
            hwfs = np.array([v['hwf'] for k, v in cameras.items()])[perm]
        c2ws = matrix_to_square(torch.tensor(c2ws)).to(device)
        # llff -> opengl
        c2ws = llff2opengl(c2ws)

        if normalized_cameras:
            ref_w2c = torch.inverse(c2ws[:1])
            c2ws = (ref_w2c.repeat(c2ws.shape[0], 1, 1)) @ c2ws[:,:,:]
            T_norm = c2ws[:, :3, 3].norm(dim=-1).max()
            c2ws[:,:3,3] = c2ws[:,:3,3] / (T_norm + 1e-2)
            #T_center = c2ws[:, :3, 3].mean(dim = 0, keepdims=True)
            points = ((torch.matmul(ref_w2c[:,:3,:3], points[:,:,None])[:,:,0] + ref_w2c[:,:3,3])) / (T_norm + 1e-2)
            # remove outliers
            in_mask = ~torch.any(abs(points) > 1.5, dim = 1)
            points = points[in_mask]
            colors = colors[in_mask]


        c2ws[:,:3,0] *= -1
        c2ws[:,:3,2] *= -1

        w2cs = torch.inverse(c2ws)
        T, R =  w2cs[:,:3,3].to(device),  w2cs[:,:3,:3].permute(0,2,1).to(device)
        focal_lengths = torch.tensor(hwfs[:,2, None] / hwfs[:,:2]).to(device)
        principal_points = torch.zeros_like(focal_lengths)
        # init cameras
        
        # load images
        cameras = PerspectiveCameras(
                focal_length=focal_lengths,
                principal_point=principal_points,
                R=R,
                T=T,
                device=device,
            )
        # vis pcd
        pcd = Pointclouds(points=[points], 
                features=[colors])
        
        # preparse scene to plot
        fig = plotly_scene_visualization(cameras.R, cameras.T, pcd = pcd, mesh =None)
        html_plot = plotly.io.to_html(fig, full_html=False, include_plotlyjs="cdn")
        # save rgb
        # s = io.BytesIO()
        # view_color_coded_images_from_tensor(images)
        # plt.savefig(s, format="png", bbox_inches="tight")
        # plt.close()
        # image_encoded = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
        HTML_TEMPLATE = """
        <html>
        <head><meta charset="utf-8"/></head>
        <body>
        {plotly_html}
        </body>
        </html>"""
        output_path = f"preview/output_scene_.html"
        with open(output_path, "w") as f:
            s = HTML_TEMPLATE.format(
                # image_encoded=image_encoded,
                # proxy_encoded =proxy_encoded,
                plotly_html=html_plot,
            )
            f.write(s)
        print('Done')