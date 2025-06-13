"""
Contains the class of DL3DV10K dataset.
"""

import os
import random
import json
import io
from PIL import Image
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from .base_dataset import MultiviewDataset
from .utils import _check_valid_rotation_matrix,  matrix_to_square
from .transformations.utils.formatting_utils import format_image

__all__ = ['DL3DV10KDataset']

class DL3DV10KDataset(MultiviewDataset):
    """ ?? """
    def __init__(self,
                 root_dir,
                 file_format='dir',
                 annotation_path = None,
                 annotation_meta=None,
                 annotation_format='pkl',
                 max_samples=-1,
                 dataset_name = 'DL3DV10K',
                 fake_length=-1,
                 debug = False,
                # dataset specific args
                 num_input_views=8,
                 num_novel_views=8,
                 sample_rate=4,
                 img_size=512,
                 normalized_cameras=True,
                 use_caption=True,
                 drop_text_p=0,
                 undistorted=True):
        # assign
        self.use_caption =use_caption
        self.img_size = img_size
        self.num_input_views  = num_input_views
        self.num_novel_views  = num_novel_views
        self.sample_rate = sample_rate
        self.normalized_cameras = normalized_cameras
        self.drop_text_p = drop_text_p
        self.undistorted = undistorted

        super().__init__(
                 root_dir = root_dir,
                 file_format=file_format,
                 annotation_path = annotation_path,
                 annotation_meta=annotation_meta,
                 annotation_format=annotation_format,
                 dataset_name=dataset_name,
                 max_samples=max_samples,
                 img_size = img_size,
                 debug=debug,
                 fake_length=fake_length)

    def build_metadata(self):
        """Build metada for dl3dv dataset if any privided"""
        items, metadata = [], {}
        example_item = '0a3e9c8e9b7713e77f45bf55edd1190c925028293aa0561feec54c826a0e6b98'
        items = os.listdir(os.path.join(self.root_dir))
        items = list(filter(lambda item: (len(item) == len(example_item)) and os.path.isdir(os.path.join(self.root_dir, item)), items))
        for item in items:
            scene_meta = dict(
                name = item,
                #filenames = filenames,
                #images = sorted(os.listdir(os.path.join(basedir, 'images_4'))),
                transforms_json = os.path.join(item, 'transforms.json')
                )
            metadata[item] = scene_meta
        items = list(metadata.keys())

        # further filter by if has captions
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

        dataset_name = self.dataset_name
        scene_name = self.items[idx]
        scene_meta = self.metadata[scene_name]
        filenames = scene_meta['filenames']

        filenames = [ff.replace('/frame', '/images_4/frame') for ff in scene_meta['filenames']]
        transforms_json = json.loads(self.fetch_file(scene_meta['transforms_json']))

        rawc2ws = dict(map(lambda x: (x['file_path'].split('/')[-1], x['transform_matrix']), transforms_json['frames']))
        tt = {ff : rawc2ws[ff.split('/')[-1]] for ff in filenames if ff.split('/')[-1] in rawc2ws.keys()}
        filenames, rawc2ws  = list(tt.keys()), list(tt.values())
        
        scene_data = dict(
            dataset_name = dataset_name,
            scene_name = scene_name,
            captions = '',
            filenames = filenames,
            num_frames = len(filenames),
            raw_c2ws =rawc2ws,
            sample_rate = self.sample_rate,
            num_input_views = self.num_input_views,
            num_novel_views = self.num_novel_views,
            transforms_json = transforms_json,
            scene_meta = scene_meta
            )
        return scene_data
    
    def get_frames_data(self,scene_data, indices, do_mirror = False, crop_pos = 0.5, flip_traj = False):
        images, c2ws, intrinsics = [], [], []
        
        filenames = scene_data['filenames']
        # frames = scene_data['frames']
        raw_c2ws = scene_data['raw_c2ws']
        transforms_json = scene_data['transforms_json']
        # scene_name = scene_data['scene_name']

        H, W = transforms_json['h'], transforms_json['w']
        Fx, Fy = transforms_json['fl_x'], transforms_json['fl_y']
        Cx, Cy = transforms_json['cx'], transforms_json['cy']
        k1, k2, p1, p2 = transforms_json['k1'], transforms_json['k2'], transforms_json['p1'], transforms_json['p2']
        affine = np.array(transforms_json['applied_transform'])

        for i in indices:
            raw_image = np.array(self.fetch_img(self.root_dir, filenames[i]))

            h, w = raw_image.shape[:2]
            l = min(w, h)
            scale = self.img_size / l
            fx, fy = Fx * h / H, Fy * w / W
            cx, cy = Cx * h / H, Cy * w / W

            # center crop
            image = self.crop(raw_image, crop_pos=crop_pos)
            image = self.transforms['resize'](image, use_dali=False)
            image = self.mirror_aug(image, do_mirror, use_dali=False)
            image = self.transforms['normalize'](image, use_dali=False)[:3]

            c2w = np.array(raw_c2ws[i])
            c2w[:3,:] = affine @ c2w # get nerfstudio -> opengl
            c2w = c2w[:3,:]
            intrinsic = np.array([fy * scale, fx * scale, 
                                    (cy - (h - l) // 2) * scale, 
                                    (cx - (w - l) // 2) * scale, 
                                    self.img_size, self.img_size])
            images.append(image)
            c2ws.append(c2w)
            intrinsics.append(intrinsic)

        images = torch.from_numpy(np.stack(images, axis=0))
        c2ws = torch.from_numpy(np.stack(c2ws, axis=0))
        intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0))

        return images, c2ws, intrinsics
