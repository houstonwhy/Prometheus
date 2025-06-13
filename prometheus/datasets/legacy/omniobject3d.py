import torch
import numpy as np
import os
import random
from PIL import Image
import random

from utils import matrix_to_square
from .base import DatasetWrapper

class OmniObject3DDataset(DatasetWrapper):
    def __init__(self, 
                 path = "/cpfs01/user/lixinyang/projects/dual3d-gs/tmp/fake_omniobject3d",
                 images_per_item  = [4, 2],
                 img_size=512,
                 max_num_scenes=-1,
                 fake_length=None,
                 normalized_cameras=False,
                 caption_path=None,
                 drop_text_p=0,
                 idfile_path=None
                 ):
        super().__init__(fake_length)

        basedirs = []

        for category in os.listdir(path):
            if not os.path.isdir(os.path.join(path, category)):
                continue
            for id in os.listdir(os.path.join(path, category)):
                basedirs.append(os.path.join(path, category, id, 'standard'))

        def filter_dataset(basedir):
            try:
                filenames = [f for f in sorted(os.listdir(os.path.join(basedir, 'images'))) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
                if len(filenames) < sum(images_per_item):
                    return False
                if not os.path.exists(os.path.join(basedir, 'cameras')):
                    return False
                return True
            except:
                return False

        basedirs = list(filter(filter_dataset, basedirs))
        
        basedirs = basedirs if max_num_scenes < 0 else basedirs[:max_num_scenes]

        # if idfile_path is not None:
        #     with open(idfile_path, 'r') as f:
        #         scenes_ids = f.readlines()
        #     scenes_ids = [scene[:-1] for scene in scenes_ids]
        #     scenes_list = list(set(scenes_list) & set(scenes_ids))

        print(f'OmniObject3D Dataset Length: {len(basedirs)}')

        self.basedirs      = basedirs
        
        self.img_size = img_size
        
        self.images_per_item  = images_per_item

        self.normalized_cameras = normalized_cameras

        if caption_path is not None:
            self.captions = {}
            import csv
            with open(caption_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    self.captions[row[0]] = row[1]
        else:
            self.captions = None

        self.drop_text_p = drop_text_p

    def __getitem__(self, index):


        index = random.randint(0, len(self.basedirs)-1)

        if True:
            basedir = self.basedirs[index]

            filenames = [f for f in sorted(os.listdir(os.path.join(basedir, 'images'))) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

            indices = np.arange(len(filenames))
            
            images = []
            c2ws = []
            intrinsics = []
            
            for n in self.images_per_item:
                random.shuffle(indices)
            
                for i in indices[:n]:

                    image = Image.open(os.path.join(basedir, 'images', filenames[i])).convert('RGB')

                    camera = np.load(os.path.join(basedir, 'cameras', filenames[i][:filenames[i].rfind('.')] + '.npy'), allow_pickle=True).item()
                    c2w, hwf = camera['c2w'], camera['hwf']

                    # center crop
                    w, h = image.size
                    l = min(w, h)
                    image = np.array(image.crop(((w - l) // 2, (h - l) // 2, w - (w - l) // 2, h - (h - l) // 2)).resize((self.img_size, self.img_size))) / 255.0
                    
                    scale = self.img_size / l
                    f = hwf[-1]
                    intrinsic = np.array([f * scale, f * scale, (h/2 - (h - l) // 2) * scale, (w/2 - (w - l) // 2) * scale, self.img_size, self.img_size])
                    images.append(image)
                    c2ws.append(c2w)
                    intrinsics.append(intrinsic)

            # -1 ~ 1
            images = torch.from_numpy(np.stack(images, axis=0)).float().permute(0,3,1,2) * 2 - 1
            c2ws = torch.from_numpy(np.stack(c2ws, axis=0))
            intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0))

            if self.normalized_cameras:
                ref_w2c = torch.inverse(matrix_to_square(c2ws[:1]))
                c2ws = (ref_w2c.repeat(c2ws.shape[0], 1, 1) @ matrix_to_square(c2ws))[:,:3,:]
            
            cameras = torch.cat([c2ws.flatten(1, 2).float(), intrinsics.float()], dim=1)
        # except:
        #     print(f"Failed to fetch data of Path: {scene_path}.")
        #     return self.__getitem__((index + 1) % len(self))

        if self.captions is None or random.random() < self.drop_text_p:
            text = ''
        else:
            try:
                text = self.captions[index] 
            except:
                text = ''

        text = text + ' 3D scene.'
        
        return images, cameras, text

    def __len__(self):
        if self.fake_length is not None:
            return self.fake_length
        else:
            return len(self.scenes_list)
        
