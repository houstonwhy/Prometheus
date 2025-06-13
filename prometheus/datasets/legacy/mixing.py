import torch
import numpy as np
import os
import random
from PIL import Image
import random
import io

from .objaverse import ObjaverseLikeDataset
from .laion import LaionDataset

class MixingDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 objaverse_dataset_kwargs={},
                 laion_dataset_kwargs={},
                 fake_length=100,
                 ):
        assert fake_length >= 1
        self.fake_length = fake_length
        self.objaverse_dataset = ObjaverseLikeDataset(**objaverse_dataset_kwargs, fake_length=fake_length)
        self.laion_dataset = LaionDataset(**laion_dataset_kwargs, fake_length=fake_length)

    def __getitem__(self, index):

        images3d, cameras, text3d = self.objaverse_dataset[index]
        images2d, text2d = self.laion_dataset[index]
        
        return images3d, cameras, text3d, images2d, text2d

    def __len__(self):
        return self.fake_length

if __name__ == "__main__":
    dataset = MixingDataset(objaverse_dataset_kwargs=dict(
        root_path='/cpfs01/user/lixinyang/datasets/objaverse_rendering_zero123/views_release',
        images_per_scene=12,
        normalized_cameras=False,
        caption_path='/cpfs01/user/lixinyang/Cap3D_automated_Objaverse.csv',
        drop_caption_p=0.1,
        idfile_path='/cpfs01/user/lixinyang/projects/objaverse-rendering/uid.txt',
    ), laion_dataset_kwargs=dict(
        img_size=256,
        root_path="/cpfs01/user/lixinyang/datasets/laion2B-en-aesthetic-data",
        drop_caption_p=0
    ))
    for i in range(len(dataset)):
        print(dataset[i])