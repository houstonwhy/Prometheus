"""
Contains the class of RealEstate10K dataset.
Code borrow from PixelSplat 
https://github.com/dcharatan/pixelsplat/blob/main/src/dataset/dataset_re10k.py
"""

import os
import gc
from io import BytesIO
import random
from copy import deepcopy
import json
import io
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms
from einops import rearrange
# import matplotlib.pyplot as plt
from prometheus.utils.camera import opencv2opengl
try:
    from .base_dataset import MultiviewDataset
    from .utils import _check_valid_rotation_matrix,  matrix_to_square
except:
    from prometheus.datasets.base_dataset import MultiviewDataset
    from prometheus.datasets.utils import _check_valid_rotation_matrix,  matrix_to_square

__all__ = ['RealEstate10KDataset','RealEstate10KDatasetEval']

# import time
# from concurrent.futures import ThreadPoolExecutor, TimeoutError

# def timeout(seconds):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             with ThreadPoolExecutor(max_workers=1) as executor:
#                 future = executor.submit(func, *args, **kwargs)
#                 try:
#                     result = future.result(timeout=seconds)
#                 except TimeoutError:
#                     print(f'Timeout: {func.__name__} took more than {seconds} seconds. Returning None.')
#                     return None
#                 return result
#         return wrapper
#     return decorator

class RealEstate10KDataset(MultiviewDataset):
    """ ?? """
    def __init__(self,
                 root_dir,
                 file_format='dir',
                 annotation_path = None,
                 annotation_meta=None,
                 annotation_format=None,
                 max_samples=-1,
                 dataset_name = 'RealEstate10K',
                 fake_length=-1,
                 debug = False,
                 scene_scale_threshold = 0.0,
                # dataset specific args
                 num_input_views=8,
                 num_novel_views=8,
                 sample_rate=4,
                 img_size=512,
                 normalized_cameras=True,
                 use_caption=False,
                 drop_text_p=0,
                 undistorted=True
                 ):
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
                 scene_scale_threshold = scene_scale_threshold,
                 dataset_name=dataset_name,
                 max_samples=max_samples,
                 img_size=img_size,
                 debug=debug,
                 fake_length=fake_length)

    def build_metadata(self):
        """Build metada for re10k dataset if not any privided"""
        items, metadata = [], {}
        # example_item = '004847.torch'
        #import ipdb ; ipdb.set_trace()
        # with open(os.path.join(self.root_dir, 'index.json'), 'r') as ff:
        #     items = json.loads(ff)
        items= eval(self.fetch_file('index.json'))
        for idx, path in items.items():
            scene_meta = dict(
                name = idx,
                path = path,
                captions = '',
                )
            metadata[idx] = scene_meta
        items = list(metadata.keys())
        #print(items[0])
        return metadata
    
    def load_torch_seq(self, seq_idx, file_path):
        raw_chunk = torch.load(os.path.join(self.root_dir, file_path), map_location='cpu', weights_only = True)
        seq_data =None
        #import ipdb; ipdb.set_trace()
        if isinstance(raw_chunk, dict):
            raw_chunk = [raw_chunk]
        for tt in raw_chunk:
            if tt['key'] == seq_idx:
                seq_data = deepcopy(tt)
                break
        #import ipdb; ipdb.set_trace()
        del raw_chunk
        #BUG slow 
        # down -> gc.collect()

        return seq_data
    def get_image(self, image_data, crop_pos = 0.5, do_mirror=False):
        use_dali = False
        # Load image
        raw_image = Image.open(BytesIO(image_data.numpy().tobytes()))
        raw_image = np.array(raw_image)
        image = self.crop(raw_image, crop_pos)
        image = self.transforms['resize'](image, use_dali=use_dali)
        image = self.mirror_aug(image, do_mirror, use_dali=use_dali)
        image = self.transforms['normalize'](image, use_dali=use_dali)[:3]
        return raw_image, image

    def parse_scene(self, idx = None, scene_name = None):
        dataset_name = self.dataset_name
        if scene_name is None:
            scene_name = self.items[idx]
        else:
            pass
        scene_meta = self.metadata[scene_name]
        scene_data = self.load_torch_seq(scene_name, file_path = scene_meta['path'])
        
        # del scene_meta
        # return dataset_name, scene_name, scene_meta, seq_data
    
        scene_data.update(
            dataset_name = dataset_name,
            scene_name = scene_name,
            captions = '',
            # filenames = filenames,
            num_frames = len(scene_data['images']),
            sample_rate = self.sample_rate,
            num_input_views = self.num_input_views,
            num_novel_views = self.num_novel_views,
            # scene_meta = scene_meta
            )
        return scene_data
    
    def get_frames_data(self, scene_data, indices, crop_pos=0.5, do_mirror=False, flip_traj=False):
        images, c2ws, intrinsics = [], [], []
        for i in indices:
            raw_image, image = self.get_image(scene_data['images'][i], crop_pos = crop_pos, do_mirror=do_mirror)
            #c2w, K = np.array(seq_data['images'], seq_data['images'])
            raw_camera = scene_data['cameras'][i]
            if isinstance(raw_camera, torch.Tensor):
                raw_camera = raw_camera.numpy()
            h, w = raw_image.shape[0:2]
            fx, fy = raw_camera[0] * w, raw_camera[1] * h
            cx, cy = raw_camera[2] * w, raw_camera[3] * h 
            w2c = np.eye(4)
            #import ipdb; ipdb.set_trace()
            w2c[:3] = rearrange(raw_camera[6:], "(h w) -> h w", h=3, w=4)
            c2w = np.linalg.inv(w2c)[:3]
            l = min(w, h)
            scale = self.img_size / l
            intrinsic = np.array(
            [fy * scale, fx * scale, 
                (cy - (h - l) // 2) * scale, (cx - (w - l) // 2) * scale, 
                self.img_size, self.img_size])
            
            intrinsics.append(intrinsic)
            c2ws.append(c2w)
            images.append(image)

        images = torch.from_numpy(np.stack(images, axis=0))
        c2ws = torch.from_numpy(np.stack(c2ws, axis=0))
        intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0))
        
        # Fro re10k OpenCV->OpenGL
        c2ws = opencv2opengl(c2ws)

        return images, c2ws, intrinsics
    



class RealEstate10KDatasetEval(RealEstate10KDataset):
        def __init__(
                self,
                 root_dir,
                 **kwargs
                ):
            super().__init__(root_dir, **kwargs)
            # print('?')

        def get_data(self, sceneid, context_ids, target_ids):
            '''
            get data from pixelsplat style eval metadata
            '''
            scene_data = self.parse_scene(scene_name = sceneid)
            # context_num = len(context_ids)

            images, c2ws, intrinsics = self.get_frames_data(
                scene_data, context_ids +target_ids)
            
            cameras = self.process_cameras(c2ws=c2ws, intrinsics=intrinsics, num_input_views = -1)

            return {
                'cameras':cameras,
                'images':images
            }

