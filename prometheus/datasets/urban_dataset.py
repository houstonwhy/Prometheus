"""
Contains the class of Urban dataset.
"""

import os
import random
import json
import io
import cv2
import numpy as np
import torch
from prometheus.utils.camera import opencv2opengl
try:
    from .base_dataset import MultiviewDataset
    from .utils import _check_valid_rotation_matrix,  matrix_to_square
    from .transformations.utils.formatting_utils import format_image
except:
    from prometheus.datasets.base_dataset import MultiviewDataset
    from prometheus.datasets.utils import _check_valid_rotation_matrix,  matrix_to_square 
    from prometheus.datasets.transformations.utils.formatting_utils import format_image
__all__ = ['UrbanGenDataset']

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

class UrbanGenDataset(MultiviewDataset):
    """ ?? """
    def __init__(self,
                 root_dir,
                 file_format=None,
                 annotation_path = None,
                 annotation_meta=None,
                 annotation_format='pkl',
                 max_samples=-1,
                 dataset_name = 'UrbanGen',
                 fake_length=-1,
                 debug = False,
                # dataset specific args
                 num_input_views=8,
                 num_novel_views=8,
                 sample_rate=4,
                 img_size=256,
                 normalized_cameras=True,
                 use_caption=True,
                 drop_text_p=0,
                 scene_scale_threshold=0.0,
                 #undistorted=True
                 ):
        # assign
        self.use_caption =use_caption
        self.img_size = img_size
        self.num_input_views  = num_input_views
        self.num_novel_views  = num_novel_views
        self.sample_rate = sample_rate
        self.normalized_cameras = normalized_cameras
        self.drop_text_p = drop_text_p
        #self.undistorted = undistorted

        super().__init__(
                 root_dir = root_dir,
                 file_format=file_format,
                 annotation_path = annotation_path,
                 annotation_meta=annotation_meta,
                 annotation_format=annotation_format,
                 dataset_name=dataset_name,
                 max_samples=max_samples,
                 img_size=img_size,
                 #use_seq_aug=False,
                 scene_scale_threshold=scene_scale_threshold,
                 debug=debug,
                 fake_length=fake_length)
        
        # For different driving dataset has different frame rate, we re-assign sample rate for each sub-dataset here
        self.sample_rate_dict = {
            'kitti' : int(sample_rate), # 10hz
            'kitti360': int(sample_rate), # 10hz
            'waymo': int(sample_rate * 2),# 10hz
            'nuscenes': int(sample_rate / 2) # 2hz
        }

    def build_metadata(self):
        """Build metada for urbangen dataset if any privided"""
        raise NotImplementedError('UrbanGenDataset only support load based on metadata ')

    
    def get_image(self, image_path, crop_pos = 0.5, do_mirror=False):
        buffer = np.frombuffer(self.fetch_file(image_path), dtype=np.uint8)
        raw_image = format_image(cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED))
        image = self.crop(raw_image, crop_pos)
        image = self.transforms['resize'](image, use_dali=False)
        image = self.mirror_aug(image, do_mirror, use_dali=False)
        image = self.transforms['normalize'](image, use_dali=False)[:3]

        return raw_image, image
    
    
    def parse_scene(self, idx):
        scene_name = self.items[idx]
        scene_meta = self.metadata[scene_name] 
        sub_dataset_name = scene_name.split('_')[0]
        dataste_name = self.dataset_name
        sample_rate = self.sample_rate_dict[sub_dataset_name]
        filenames = [frame_meta['image_path'] for frame_meta in scene_meta]
    
        scene_data = dict(
            dataset_name = dataste_name,
            scene_name = scene_name,
            num_frames = len(filenames),
            filenames = filenames,
            scene_meta = scene_meta,
            sample_rate = sample_rate,
            num_input_views = self.num_input_views,
            num_novel_views = self.num_novel_views
            )
            
        return scene_data

    def get_frames_data(self,scene_data, indices, do_mirror = False, crop_pos = 0.5, flip_traj = False):

        filenames, scene_meta = scene_data['filenames'], scene_data['scene_meta']
        images, c2ws, intrinsics = [], [], []
        # Get full data frame-by-frame
        for i in indices:
            raw_image, image = self.get_image(filenames[i], crop_pos = crop_pos, do_mirror=do_mirror)
            c2w, K = np.array(scene_meta[i]['cam2world'])[:3,:], np.array(scene_meta[i]['cam_K'])
            #fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            h, w = raw_image.shape[0:2]
            fx, fy = K[0,0], K[1,1]
            cx, cy = w / 2, h / 2
            l = min(w, h)
            scale = self.img_size / l
            intrinsic = np.array(
            [fy * scale, fx * scale, 
             (cy - (h - l) // 2) * scale, (cx - (w - l) // 2) * scale, 
             self.img_size, self.img_size])
            images.append(image)
            c2ws.append(c2w)
            intrinsics.append(intrinsic)

        images = torch.from_numpy(np.stack(images, axis=0))
        c2ws = torch.from_numpy(np.stack(c2ws, axis=0))
        intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0))
        c2ws = opencv2opengl(c2ws)

        return images, c2ws, intrinsics
    
    
