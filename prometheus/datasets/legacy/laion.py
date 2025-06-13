import webdataset as wds
import torch
import numpy as np
import os
import random
from PIL import Image
import random
import io
import json
import hashlib

from torchvision import transforms 
from .base import DatasetWrapper

class LaionDataset(DatasetWrapper):
    def __init__(self, 
                 img_size=256,
                 root_path="/cpfs01/user/lixinyang/datasets/laion2B-en-aesthetic-data",
                 fake_length=100,
                 images_per_iter=1,
                 drop_text_p=0,
                 error_image_path="/cpfs01/user/lixinyang/projects/dual3d-gs/tmp/laion_error_test/example.jpg",
                 ):
        super().__init__(fake_length)

        urls = sorted([os.path.join(root_path, filename) for filename in os.listdir(root_path) if filename[-3:] == 'tar'])# [:100]
        dataset = wds.WebDataset(urls, shardshuffle=True, cache_dir=None, cache_size=10 ** 10, handler=wds.handlers.warn_and_continue, resampled=True, nodesplitter=wds.shardlists.split_by_node)

        def filter_dataset(item):
            
            if 'txt' not in item:
                return False
            if 'jpg' not in item:
                return False
            if 'json' not in item:
                return False
            else:
                meta = json.loads(item['json'].decode("utf-8"))
                if 'status' not in meta or meta['status'] != 'success':
                    return False

            return True

        filtered_dataset = dataset.select(filter_dataset)

        if error_image_path is not None:
            self.to_tensor = transforms.ToTensor()
            self.error_image = self.to_tensor(Image.open(error_image_path).convert('RGB'))
            self.error_image_h, self.error_image_w = self.error_image.shape[1:]
        else:
            self.error_image = None

        transform = transforms.Compose(
            [
                transforms.Resize(280),
                transforms.RandomCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        def preprocess_dataset(item):
            output = {}

            output['is_error'] = False

            if True:
                image_data = item['jpg']
                image = Image.open(io.BytesIO(image_data))
                w, h = image.size
                output['key'] = item['__key__']

                if self.error_image is not None:
                    if h == self.error_image_h and w == self.error_image_w:
                        original_image = self.to_tensor(image)
                        if (original_image - self.error_image).abs().mean() < 0.01:
                            output['is_error'] = True
                            return output

                image_tensor = transform(image)
                output["image_tensor"] = image_tensor
                image = output['image_tensor']
                           
            if True:
                text = item['txt']
                caption = text.decode("utf-8")
                output["text"] = caption

            return output

        self.transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_continue)

        self.drop_text_p = drop_text_p
        self.iter = iter(self.transformed_dataset)

        self.images_per_iter = images_per_iter

    def __getitem__(self, index):
        images, texts = [], [] 

        for i in range(self.images_per_iter):
            while True:
                data = next(self.iter)
                if data['is_error']:
                    pass
                    # print(f"[LAION] skip {data['key']}")
                else:
                    break

                
            image, text = data["image_tensor"], data['text']

            if random.random() < self.drop_text_p:
                text = ''
            
            images.append(image)
            texts.append(text)
        
        images = torch.stack(images, dim=0)
        return images, texts