import torch
import numpy as np
import os
import random
from PIL import Image

import trimesh

def visualize_poses(poses, size=0.1, save_dir="./"):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    # trimesh.Scene(objects).show()
    trimesh.Scene(objects).export(os.path.join(save_dir,"test.glb"))


def blender_matrix_to_ngp(pose):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3]],
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3]],
    ], dtype=np.float32)
    return new_pose

class ShapenetDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 images_per_item  = [4, 2],
                 images_per_scene = 24,
                 lens=560,
                 img_size=512,
                 root_path = "/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/lxy/datasets/shapenet/shapenet_cars",
                 max_num_scenes=-1,
                 ):
        
        assert sum(images_per_item) <= images_per_scene
        
        scenes_list = os.listdir(root_path)
        scenes_list = scenes_list if max_num_scenes < 1 else scenes_list[:max_num_scenes]
        self.scenes_list      = scenes_list
        self.root_path        = root_path
        
        self.lens = lens
        self.img_size = img_size
        
        self.intrinsic = [lens, lens, img_size/2, img_size/2]
        
        self.images_per_item  = images_per_item
        self.images_per_scene = images_per_scene

    def __getitem__(self, index):
        try:
            scene_path = os.path.join(self.root_path, self.scenes_list[index])
            images_list = np.arange(self.images_per_scene)
            
            images = []
            c2ws = []
            
            for n in self.images_per_item:
                random.shuffle(images_list)
            
                for i in images_list[:n]:
                    images.append(np.array(Image.open(os.path.join(scene_path, f"{i:03}.png")).convert('RGB')))
                    w2c = np.load(os.path.join(scene_path, f"{i:03}.npy"))
                    
                    w2c = w2c
                    R = w2c[:3,:3]
                    R_inv = R.T
                    t = - R_inv @ w2c[:3,3]
                    c2w = np.column_stack([R_inv, t])
                    
                    c2ws.append(c2w)
            
            # -1 ~ 1
            images = torch.from_numpy(np.stack(images, axis=0)).permute(0,3,1,2) / 127.5 - 1
            c2ws = torch.from_numpy(np.stack(c2ws, axis=0)) 
            
            cameras = torch.cat([c2ws.flatten(1, 2).float(), torch.Tensor([*self.intrinsic, self.img_size, self.img_size])[None].repeat(sum(self.images_per_item), 1)], dim=1)
        except:
            print(f"Failed to fetch data of Path: {scene_path}.")
            return self.__getitem__((index + 1) % len(self))
        
        return images, cameras, 'a 3D car in shapenet dataset.'

    def __len__(self):
        return len(self.scenes_list)
        
