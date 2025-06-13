import torch
import numpy as np
import csv
import os
import os.path as osp
import json
import random
from PIL import Image

# from utils import matrix_to_square
from .base_dataset import MultiviewDataset
from prometheus.utils.camera import opencv2opengl
# from .utils import orbit_camera_jitter

class ObjaverseDataset(MultiviewDataset):
    def __init__(self, 
                root_dir ,
                metadata_dir,
                # images_per_item  = [4, 2],
                # images_per_scene = 12,
                use_caption=True,
                img_size=256,
                max_samples=-1,
                debug= False,
                fake_length=-1,
                normalized_cameras=False,
                # caption_path=None,
                images_per_scene=16,
                drop_text_p=0,
                orbit_camera_jitter_p=0,
                view_type='random',#'offset0'
                idfile_path='_v0.txt',
                captions_filename="cap3d.csv",
                prompt_suffix="",
                prompt_prefix="[3D Asset]",
                dataset_name="Objaverse",
                num_input_views = 1,
                num_novel_views = 1,
                sample_rate = 1,
                ):
        # super().__init__(fake_length)

        # assert max(images_per_item) <= images_per_scene
        self.metadata_dir = metadata_dir
        self.num_novel_views  = num_novel_views
        self.num_input_views  = num_input_views
        self.drop_text_p = drop_text_p
        self.orbit_camera_jitter_p = orbit_camera_jitter_p
        self.use_caption =use_caption
        self.img_size = img_size
        self.sample_rate = sample_rate
        self.normalized_cameras = normalized_cameras

        self.images_per_scene = images_per_scene
        self.prompt_suffix = prompt_suffix
        self.prompt_prefix = prompt_prefix

        # self.view_type = view_type
        self.caption_path = osp.join(metadata_dir, captions_filename)
        self.idfile_path = osp.join(metadata_dir, idfile_path)

        # scenes_list = os.listdir(root_path)
        # scenes_list = scenes_list if max_num_scenes < 0 else scenes_list[:max_num_scenes]
        
        # self.intrinsic = [lens, lens, img_size/2, img_size/2]
        
        # self.images_per_item  = images_per_item
        # self.images_per_scene = images_per_scene
        
        super().__init__(
                 root_dir = root_dir,
                 file_format='dir',
                 annotation_path = None,
                 annotation_meta=None,
                 annotation_format=None,
                 dataset_name=dataset_name,
                 max_samples=max_samples,
                 img_size = img_size,
                 view_type = view_type,
                 debug=debug,
                 fake_length=fake_length)
    
    def build_metadata(self):
        """Build metadata for Objaverse dataset"""
        # items, metadata = [], {}
        # yyb/datasets/objaverse_render/_v0/000-024/0a2a571ccd4f48a6b78dd65e6e668890/random/renderings
        caption_path, idfile_path = self.caption_path, self.idfile_path
        full_items = [f'{x.strip()}' for x in open(self.idfile_path)]
        if 'data1' in self.root_dir:# for local workstation, only keep tiny subset that store in local 
            items = [i for i in full_items if osp.exists(osp.join(self.root_dir, '_v0', i,'uniform/renderings/00000000_rgba.png'))]
            print('Filtering valid paths...')
        else:
            items = full_items
        # if self.max_samples != -1:
        #     self.paths = self.paths[:self.max_samples]
        # print('Loaded image dirs of Objaverse.')
        # print('jiahao debug', len(self.paths))

        # caption_path = self.captions_filename
        # print(f'Loading captions from {caption_path}...')
        caption_dict = {}
        with open(self.caption_path, 'r') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                # line = line.strip().split(',')
                caption_dict[line[0]] = ','.join(line[1:])
        # print('Loaded captions.')
        self.captions = {x : caption_dict[x.split('/')[-1]] for x in items if x.split('/')[-1] in caption_dict.keys()}
        # self.items = list(self.captions.keys())
        # print(f'Objeaverse Number {len(self.captions)} of objects')
        return self.captions
        
        
    def parse_scene(self, idx):
        dataset_name = self.dataset_name
        scene_name = self.items[idx]
        caption = self.metadata[scene_name]
        # scene_name = '_'.join(scene_name.split('/')[-2:])
        scene_dir =osp.join('_v0',scene_name, 'uniform')
        json_path =osp.join(scene_dir, 'opencv_cameras.json')
        # image_filenames = [osp.join('_v0',scene_name, '' ,f'{view_idx:08d}_rgba.png') for view_idx in range(self.images_per_scene)]
        scene_meta =  json.loads(self.fetch_file(json_path))['frames']
        if  len(scene_meta) < 8:
            raise Value(f'scene has only {len(scene_meta)} frames')
        num_input_views, num_novel_views = self.num_input_views, self.num_novel_views
        sample_rate = -1 if self.view_type == 'random' else self.sample_rate 

        scene_data = dict(
            scene_dir = scene_dir,
            num_frames = len(scene_meta),
            scene_meta = scene_meta,
            dataset_name = dataset_name,
            scene_name = '_'.join(scene_name.split('/')[-2:]),
            caption = caption,
            num_input_views=num_input_views,
            num_novel_views=num_novel_views,
            sample_rate = sample_rate
            )
        return scene_data
    
    def get_caption(self, scene_data):
        if not self.use_caption or random.random() < self.drop_text_p:
            text = 'XXX'
        else:
            text = scene_data['caption']
        text = self.prompt_prefix + text + self.prompt_suffix
        return text
    

    def get_frames_data(self,scene_data, indices, do_mirror = False, crop_pos = 0.5, flip_traj = False):
        images, c2ws, intrinsics = [], [], []
        scene_meta, scene_dir = scene_data['scene_meta'], scene_data['scene_dir']
        for i in indices:
            filename = osp.join(scene_dir, scene_meta[i]['file_path'])
            
            # camera = cameras[str(i)]
            # c2w, hwf = np.array(camera['c2w']), np.array(camera['hwf'])
            raw_image = np.array(self.fetch_img(self.root_dir, filename))
            fx, fy, cx, cy = scene_meta[i]['fx'], scene_meta[i]['fy'], scene_meta[i]['cx'], scene_meta[i]['cy']
            c2w = np.linalg.inv(scene_meta[i]['w2c'])[:3]
            h, w = raw_image.shape[:2]
            l = min(w, h)
            scale = self.img_size / l

            c2w = c2w[:3,:]
            intrinsic = np.array([fy * scale, fx * scale, 
                                    (cy - (h - l) // 2) * scale, 
                                    (cx - (w - l) // 2) * scale, 
                                    self.img_size, self.img_size])
            # center crop
            image = self.crop(raw_image, crop_pos=crop_pos)
            image = self.transforms['resize'](image, use_dali=False)
            image = self.mirror_aug(image, do_mirror, use_dali=False)
            image = self.transforms['normalize'](image, use_dali=False)[:3]

            
            intrinsics.append(intrinsic)
            c2ws.append(c2w)
            images.append(image)

        images = torch.from_numpy(np.stack(images, axis=0))
        c2ws = torch.from_numpy(np.stack(c2ws, axis=0))
        intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0))
        
        #c2ws = matrix_to_square(c2ws)
        c2ws = opencv2opengl(c2ws)

        return images, c2ws, intrinsics
    
    # # @timeout(100)
    # def get_raw_data(self, idx):
    #     # 1.Parse sequence  (dataset specific)
    #     scene_data = self.parse_scene(idx)

    #     dataset_name, scene_name = scene_data['dataset_name'], scene_data['scene_name']

    #     # 2.Get Caption (general)
    #     text = self.get_caption(scene_data=scene_data)
    #     if self.use_caption and (text is None):
    #         return None

    #     # 3.Sample views
    #     full_indices = self.sample_views(
    #         num_frames=scene_data['num_frames'],
    #         sample_rate=scene_data['sample_rate'],
    #         num_input_views=scene_data['num_input_views'],
    #         num_novel_views=scene_data['num_novel_views']
    #     )
    #     # if full_indices is None:
    #     #     return None

    #     # 4. Get full data frame-by-frame (dataset specific)
    #     #TODO add more image / trajectory augmentation
    #     do_mirror, crop_pos, flip_traj = False, 0.5, False
    #     images, c2ws, intrinsics = self.get_frames_data(
    #         scene_data = scene_data, 
    #         indices=full_indices,
    #         do_mirror = do_mirror, 
    #         crop_pos = crop_pos, 
    #         flip_traj = flip_traj
    #         )
        
    #     # 5. Pose Normalization (general)
    #     cameras = self.process_cameras(c2ws=c2ws, intrinsics=intrinsics)
    #     if cameras is None:
    #         return None

    #     return [images, cameras, text, dataset_name, scene_name]
    