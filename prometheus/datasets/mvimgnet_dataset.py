"""
Contains the class of MVImgNet dataset.
"""
import os
import random
import io
from PIL import Image

import torch
import numpy as np

from .base_dataset import MultiviewDataset
from .utils import matrix_to_square
from prometheus.utils.camera import llff2opengl

class MVImgNetDataset(MultiviewDataset):
    """XX"""
    def __init__(self,
                 root_dir,
                 file_format = 'dir',
                 annotation_path = None,
                 annotation_meta=None,
                 annotation_format='pkl',
                 max_samples = -1,
                 fake_length = -1,
                 dataset_name = 'MVImgNet',
                 debug = False,
                 category = 'all',
                 num_input_views=8,
                 num_novel_views=8,
                 sample_rate=3,
                 img_size=512,
                 normalized_cameras=True,
                 use_caption=True,
                 drop_text_p=0
                 ):
        """XX"""

        self.category = category
        self.num_input_views=num_input_views
        self.num_novel_views=num_novel_views
        self.sample_rate=sample_rate
        self.img_size=img_size
        self.normalized_cameras=normalized_cameras
        self.use_caption=use_caption
        self.drop_text_p=drop_text_p

        super().__init__(
                 root_dir = root_dir,
                 file_format=file_format,
                 annotation_path = annotation_path,
                 annotation_meta=annotation_meta,
                 annotation_format=annotation_format,
                 max_samples=max_samples,
                 dataset_name=dataset_name,
                 img_size = img_size,
                 debug=debug,
                 fake_length=fake_length)


    def build_metadata(self):
        """Build metada for mvimgnet dataset if any privided"""
        items, metadata = [], {}
        #example_item = '0a3e9c8e9b7713e77f45bf55edd1190c925028293aa0561feec54c826a0e6b98'
        # 0. filter categories
        category_list = []
        if self.debug:
            category_list += ['0']
        elif self.category == 'all':
            category_list = os.listdir(self.root_dir)
        else:
            category_list += [self.category ]
        # 1. filter calid scenes
        for cate in category_list:
            cate_items = [os.path.join(cate, i) for i in os.listdir(os.path.join(self.root_dir, cate))]
            for item in cate_items:
                filenames = [os.path.join(item, 'images', f) for f in sorted(os.listdir(os.path.join(self.root_dir, item, 'images'))) if f.endswith('jpg')]
                if len(filenames) < (self.num_input_views - 1) * self.sample_rate + 1:
                    continue
                if not os.path.exists(os.path.join(self.root_dir, item, 'cameras.json')):
                    continue
                # add item in metadata
                scene_meta = dict(
                    name = item,
                    filenames = filenames,
                    #images = sorted(os.listdir(os.path.join(basedir, 'images_4'))),
                    camerasjson = os.path.join(item, 'cameras.json')
                    )
                metadata[item] = scene_meta
        items = list(metadata.keys())

        # 2. further filter by if has captions
        if self.use_caption:
            for item in items:
                if os.path.exists(os.path.join(self.root_dir, item, 'captions.txt')):
                    # self.captions[basedir] = []
                    with open(os.path.join(self.root_dir, item, 'captions.txt'), 'r', encoding='utf-8')as f:
                        rows = f.readlines()
                    #captions[basedir] = [row.replace('\n', '') for row in rows]
                    metadata[item]['captions'] = [row.replace('\n', '') for row in rows]
            metadata = {k :v for k, v in metadata.items() if 'captions' in v.keys()}
        return metadata


    def parse_scene(self, idx):
        #idx = random.randint(0, len(self.items)-1)
        dataset_name = self.dataset_name
        scene_name = self.items[idx]
        scene_meta = self.metadata[scene_name]
        scene_name = '_'.join(scene_name.split('/')[-2:])
        
        # sample_rate = self.sample_rate
        filenames = scene_meta['filenames']
        filenames.sort()
        
        if ('cameras' in scene_meta.keys())  and scene_meta['cameras']:
            cameras = scene_meta['cameras']
            perm = np.array(([int(k) for k in cameras.keys()]))
            #Notice that perm = np.argsort(names) rather than frame index
            cameras_c2w = np.array([v['c2w'] for k, v in cameras.items()])[perm]
            cameras_hwf = np.array([v['hwf'] for k, v in cameras.items()])[perm]
        else:
            raise ValueError('No cameras')
        if not len(filenames) == len(cameras):
            raise ValueError('filenames not consist with cameras')
        #assert len(filenames) == len(cameras)
        scene_data = dict(
            dataset_name = dataset_name,
            scene_name = scene_name,
            filenames = filenames,
            sample_rate = self.sample_rate,
            num_input_views = self.num_input_views,
            num_novel_views = self.num_novel_views,
            num_frames = len(filenames),
            cameras_c2w = cameras_c2w,
            cameras_hwf = cameras_hwf,
            scene_meta = scene_meta)
        return scene_data
    
    def get_frames_data(self,scene_data, indices, do_mirror = False, crop_pos = 0.5, flip_traj = False):
        images, c2ws, intrinsics = [], [], []
        filenames, cameras_c2w, cameras_hwf = scene_data['filenames'], scene_data['cameras_c2w'], scene_data['cameras_hwf']
        for i in indices:
            raw_image, image =self.get_image(filenames[i], crop_pos=0.5)
            # mvimgenet use h/2 and w/2 as cx, cy
            c2w, hwf = cameras_c2w[i], cameras_hwf[i]
            # w, h = raw_image.shape[:2]
            h, w = raw_image.shape[:2]
            l = min(w, h)
            scale = self.img_size / l
            f = hwf[-1]
            intrinsic = np.array([f * scale, f * scale, (h/2 - (h - l) // 2) * scale, (w/2 - (w - l) // 2) * scale, self.img_size, self.img_size])
            images.append(image)
            c2ws.append(c2w)
            intrinsics.append(intrinsic)
        # # -1 ~ 1
        # images = torch.from_numpy(np.stack(images, axis=0)).float().permute(0,3,1,2) * 2 - 1
        images = torch.from_numpy(np.stack(images, axis=0))
        intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0))
        c2ws = torch.from_numpy(np.stack(c2ws, axis=0))
        #c2ws = matrix_to_square(c2ws)
        c2ws = llff2opengl(c2ws)

        return images, c2ws, intrinsics
