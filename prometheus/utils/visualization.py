"""Visualization related utils
"""
#TODO
from typing import NamedTuple
import base64
from PIL import Image
import io
import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt

import plotly
import plotly.io as pio
import kaleido #noqa
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer.cameras import FoVPerspectiveCameras, look_at_view_transform
from pytorch3d.vis.plotly_vis import plot_scene


__all__ = ['plotly_scene_visualization']

HTML_TEMPLATE = """<html><head><meta charset="utf-8"/></head>
<body><img src="data:image/png;charset=utf-8;base64, {image_encoded}"/>
{plotly_html}</body></html>"""

class AxisArgs(NamedTuple):  # pragma: no cover
    showgrid: bool = False
    zeroline: bool = False
    showline: bool = False
    ticks: str = ""
    showticklabels: bool = False
    backgroundcolor: str = "#fff"
    showaxeslabels: bool = False

def get_w2c(camera):
    w2c = np.linalg.inv(camera)
    return w2c

def get_pytorch3d_cameras(camera_file):
    camera_scene=[]
    if isinstance(camera_file, (str)):
        cameras = torch.from_numpy(np.load(camera_file))
    elif isinstance(camera_file, (torch.Tensor)):
        cameras = camera_file
    elif isinstance(camera_file, (np.ndarray)):
        cameras = torch.from_numpy(camera_file)
    else:
        raise ValueError('Unsupport input type')
    num_frames = cameras.shape[0]
    camera = {}
    for i in range(0,num_frames,1):
        c2w = torch.eye(4)
        c2w[:3,:] =cameras[i][:12].reshape(3,4)
        fx,fy,cx,cy,H,W = cameras[i][12:].chunk(6,-1)
        K = [
                    [fx,   0,   cx,   0],
                    [0,   fy,   cy,   0],
                    [0,    0,    0,   1],
                    [0,    0,    1,   0],
            ]
        K = torch.tensor(K).unsqueeze(0)
        #fov = 2 * np.arctan2(H / 2, K[0, 0, 0]) 
        c2w[:3,0] *= -1
        c2w[:3,2] *= -1
        
        w2c = get_w2c(c2w)
        R, T = w2c[None,:3,:3].transpose(0,2,1) ,w2c[None,:3,3] # R to row major B, Inspired by: https://github.com/guanyingc/pytorch3d_render_colmap/blob/master/render_colmap_mesh.py#L150
        camera[i] = PerspectiveCameras(R=R, T=T,K=K) 
    return camera

def plotly_scene_visualization(camera_file, pcd = None, key_frame_rate: int = 1, img_return: bool = False,  cmap: str = "Spectral", **cam_keargs):
    """Visualize camera trajectory with scene pcd (optional)
    """
    camera_scene=[]
    if isinstance(camera_file, (str)):
        cameras = torch.from_numpy(np.load(camera_file))
    elif isinstance(camera_file, (torch.Tensor)):
        cameras = camera_file
    elif isinstance(camera_file, (np.ndarray)):
        cameras = torch.from_numpy(camera_file)
    else:
        raise ValueError('Unsupport input type')
    num_frames = cameras.shape[0]
    camera = {}
    for i in range(0,num_frames,key_frame_rate):
        c2w = torch.eye(4)
        c2w[:3,:] =cameras[i][:12].reshape(3,4)
        fx,fy,cx,cy,H,W = cameras[i][12:].chunk(6,-1)
        K = [
                    [fx,   0,   cx,   0],
                    [0,   fy,   cy,   0],
                    [0,    0,    0,   1],
                    [0,    0,    1,   0],
            ]
        K = torch.tensor(K).unsqueeze(0)
        #fov = 2 * np.arctan2(H / 2, K[0, 0, 0]) 
        c2w[:3,0] *= -1
        c2w[:3,2] *= -1
        
        w2c = get_w2c(c2w)
        R, T = w2c[None,:3,:3].transpose(0,2,1) ,w2c[None,:3,3] # R to row major B, Inspired by: https://github.com/guanyingc/pytorch3d_render_colmap/blob/master/render_colmap_mesh.py#L150
        camera[i] = PerspectiveCameras(R=R, T=T,K=K) 
            
    camera_scene.append(camera)
    # for visual convinience
    array_axs =[1, 0.5, 0,-0.5, -1] 
    array_axs = [x * 1.5 for x in array_axs]
    range_axs = [-1, 1]
    range_axs = [x * 1.5 for x in range_axs]
    
    dist = cam_keargs.get('dist', -1.5)
    elev = cam_keargs.get('elev', -52)
    azim = cam_keargs.get('azim', 180)

    # demo default view transform 
    R, T = look_at_view_transform(dist, elev, azim)

    geo_to_plot = {**camera_scene[0]}
    if pcd is not None:
        geo_to_plot.update({"pcd":pcd})
        
        array_axs =[1, 0.5, 0,-0.5, -1] 
        array_axs = [x * 5 for x in array_axs]
        range_axs = [-1, 1]
        range_axs = [x * 5 for x in range_axs]
        # dist = -2
        # elev = -30   
        # azim = 180 
    # craeate view camera
    #cameras_view = FoVPerspectiveCameras(R=R, T=T)
    cameras_view = PerspectiveCameras(R=R, T=T)
    fig = plot_scene(
        {"scene":geo_to_plot},
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
        camera_scale=0.15,
        pointcloud_marker_size=1.2,
        # axis_args=AxisArgs(showline=False,showgrid=True,zeroline=True,showticklabels=False,showaxeslabels=True),
        viewpoint_cameras=cameras_view,
    )

    # cmap = plt.get_cmap("Spectral")
    cmap = plt.get_cmap(cmap)
    for i in range(len(fig.data)):
        fig.data[i].line.color = matplotlib.colors.to_hex(cmap(i/ (num_frames // key_frame_rate)))
    fig.update_layout(showlegend=False)
    
    if img_return:
        return Image.open(io.BytesIO(pio.to_image(fig,width=800, height=800)))
        #return plotly.io.write_image(fig, 'aa.png', width=800, height=800)
    else:
        return fig