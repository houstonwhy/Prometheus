import torch
import numpy as np
import os
import random
from PIL import Image
import random

from utils import matrix_to_square
from .base import DatasetWrapper

from .utils import orbit_camera_jitter

class ObjaverseLikeDataset(DatasetWrapper):
    def __init__(self, 
                 images_per_item  = [4, 2],
                 images_per_scene = 12,
                 lens=560,
                 img_size=512,
                 root_path = "/cpfs01/user/lixinyang/datasets/views_release",
                 max_num_scenes=-1,
                 fake_length=None,
                 normalized_cameras=False,
                 caption_path=None,
                 drop_text_p=0,
                 orbit_camera_jitter_p=0,
                 idfile_path=None,
                 ):
        super().__init__(fake_length)

        assert max(images_per_item) <= images_per_scene
        
        scenes_list = os.listdir(root_path)
        scenes_list = scenes_list if max_num_scenes < 0 else scenes_list[:max_num_scenes]

        if idfile_path is not None:
            with open(idfile_path, 'r') as f:
                scenes_ids = f.readlines()
            scenes_ids = [scene[:-1] for scene in scenes_ids]
            scenes_list = list(set(scenes_list) & set(scenes_ids))

        print(f'Objaverse Dataset Length: {len(scenes_list)}')

        self.scenes_list      = scenes_list
        self.root_path        = root_path
        
        self.lens = lens
        self.img_size = img_size
        
        self.intrinsic = [lens, lens, img_size/2, img_size/2]
        
        self.images_per_item  = images_per_item
        self.images_per_scene = images_per_scene

        self.normalized_cameras = normalized_cameras

        if caption_path is not None:
            self.captions = {}
            if caption_path.endswith('csv'):
                import csv
                with open(caption_path, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        self.captions[row[0]] = [row[1]]
            elif caption_path.endswith('json'):
                import json
                with open(caption_path, 'r') as f:
                    rows = json.load(f)
                    for row in rows:
                        self.captions[row["obj_id"]] = [row["cap3d"], row["3dtopia"]]
            else:
                assert False, f'Caption file {caption_path} is not supported!'
        else:
            self.captions = None

        self.drop_text_p = drop_text_p
        self.orbit_camera_jitter_p = orbit_camera_jitter_p

    def __getitem__(self, index):

        index = random.randint(0, len(self.scenes_list)-1)

        try:
            scene_path = os.path.join(self.root_path, self.scenes_list[index])
            images_list = np.arange(self.images_per_scene)
            
            images = []
            c2ws = []
            
            for n in self.images_per_item:
                random.shuffle(images_list)
            
                for i in images_list[:n]:
                    image = np.array(Image.open(os.path.join(scene_path, f"{i:03}.png")).convert('RGBA')) / 255
                    # white background
                    image = image[..., :3] * image[..., 3:] + (1 - image[..., 3:]) * 1
                    images.append(image)
                    w2c = np.load(os.path.join(scene_path, f"{i:03}.npy"))
                    
                    R = w2c[:3,:3]
                    R_inv = R.T
                    t = - R_inv @ w2c[:3,3]
                    c2w = np.column_stack([R_inv, t])

                    # c2w[:3, 1:3] *= -1
                    
                    c2ws.append(c2w)
            
            # -1 ~ 1
            images = torch.from_numpy(np.stack(images, axis=0)).float().permute(0,3,1,2) * 2 - 1
            c2ws = torch.from_numpy(np.stack(c2ws, axis=0)) 

            if random.random() < self.orbit_camera_jitter_p:
                c2ws[1:] = orbit_camera_jitter(matrix_to_square(c2ws[1:]))[:,:3,:]

            if self.normalized_cameras:

                ref_w2c = torch.inverse(matrix_to_square(c2ws[:1]))
                c2ws = (ref_w2c.repeat(c2ws.shape[0], 1, 1) @ matrix_to_square(c2ws))[:,:3,:]
                # R_inv = c2ws[:1, :3, :3].transpose(-1, -2)
                # c2ws[:, :3, :3] = R_inv @ c2ws[:, :3, :3]
                # c2ws[:, :3, 3:] = R_inv @ c2ws[:, :3, 3:] 
                # c2ws[:, :3, 3:] -= c2ws[:1, :3, 3:]
            
            cameras = torch.cat([c2ws.flatten(1, 2).float(), torch.Tensor([*self.intrinsic, self.img_size, self.img_size])[None].repeat(sum(self.images_per_item), 1)], dim=1)
        except:
            print(f"Failed to fetch data of Path: {scene_path}.")
            return self.__getitem__((index + 1) % len(self))

        if self.captions is None or random.random() < self.drop_text_p:
            text = ''
        else:
            try:
                if type(self.captions[self.scenes_list[index]]) is str:
                    text = self.captions[self.scenes_list[index]]
                elif type(self.captions[self.scenes_list[index]]) is list or type(self.captions[self.scenes_list[index]]) is tuple:
                    text = random.choice(self.captions[self.scenes_list[index]])
                else:
                    text = ''
            except:
                text = ''

        text = text + ' 3D asset.'
        
        return images, cameras, text

    def __len__(self):
        if self.fake_length is not None:
            return self.fake_length
        else:
            return len(self.scenes_list)
        
