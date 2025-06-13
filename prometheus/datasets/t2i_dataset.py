"""
Contains the class of Text2Image dataset e.g. SAM1B, LAION5B. 
"""
import os
import pickle
import random
import csv
import json
from PIL import Image
import torch
import numpy as np
import pandas as pd

from .base_dataset import BaseDataset


__all__ = ['Text2ImageDataset', 'Text2ImageDatasetV1']


class _TorchSerializedList(object):
    """
    A list-like object whose items are serialized and stored in a torch tensor. When
    launching a process that uses TorchSerializedList with "fork" start method,
    the subprocess can read the same buffer without triggering copy-on-access. When
    launching a process that uses TorchSerializedList with "spawn/forkserver" start
    method, the list will be pickled by a special ForkingPickler registered by PyTorch
    that moves data to shared memory. In both cases, this allows parent and child
    processes to share RAM for the list data, hence avoids the issue in
    https://github.com/pytorch/pytorch/issues/13246.

    See also https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
    on how it works.
    """

    def __init__(self, txt_file):

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if isinstance(txt_file, list):
            self._lst = [_serialize(x) for x in txt_file]
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        else:
            self._lst = []
            self._addr = []
            txt_list = txt_file.split(',')
            #self._lst = [_serialize(x) for x in self._lst]
            #self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            for single_txt in txt_list:
                print(f'loading from {single_txt}')
                with open(single_txt, 'r') as f:
                    line = f.readline() # skip the first row
                    while line:
                        line = f.readline().strip()
                        serialized_line = _serialize(line)
                        self._lst.append(serialized_line)
                        self._addr.append(len(serialized_line))
                    f.close()
                    self._lst.pop()
                    self._addr.pop()
            self._addr = np.asarray(self._addr, dtype=np.int64)
        self._addr = torch.from_numpy(np.cumsum(self._addr))
        self._lst = torch.from_numpy(np.concatenate(self._lst))
        print(("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2)))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())

        # @lint-ignore PYTHONPICKLEISBAD
        return pickle.loads(bytes)

def read_txt(txt_file, serialize=True):
    if txt_file is None:
        return None

    if serialize:
        return _TorchSerializedList(txt_file)

    if isinstance(txt_file, str) and ',' in txt_file:
        txt_paths = txt_file.split(',')
    else:
        if not isinstance(txt_file, list):
            txt_paths = [txt_file]
        else:
            txt_paths = txt_file
    txt_list = []
    for txt_path in txt_paths:
        with open(txt_path, 'r') as f:
            line = f.readline() # skip the first row
            while line:
                line = f.readline().strip()
                txt_list.append(line)
            f.close()
            txt_list.pop()
    return txt_list

def read_csv(csv_file, serialize=True):
    img_list = []
    filepath_list = []
    caption_list = []
    if isinstance(csv_file, str) and ',' in csv_file:
        csv_paths = csv_file.split(',')
    else:
        if not isinstance(csv_file, list):
            csv_paths = [csv_file]
        else:
            csv_paths = csv_file

    for csv_file in csv_paths:
        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f)
            for i, line in enumerate(csv_reader):
                if i == 0:
                    continue
                if '.jpg' not in line[0]:
                    continue
                img_list.append(line[0])
                filepath_list.append(line[1])
                caption_list.append(line[2])

    if serialize:
        img_list = _TorchSerializedList(img_list)
        caption_list = _TorchSerializedList(caption_list)
    return img_list, filepath_list, caption_list

class Text2ImageDataset(BaseDataset):
    """
    Defines the Text2Image dataset class.
    """
    def __init__(self,
                 root_dir,
                 img_txt = None,
                 text_txt = None,
                 csv_file=None,
                 resolution=64,
                 images_per_iter=1,
                 ms_training_res=[],
                 min_val=-1,
                 max_val=1,
                 image_channels=3,
                 cfg_prob=0.0,
                 dataset_name='SAM_1B',
                 serialize=False,
                 fake_length=-1,
                 max_samples=-1,
                 debug = False
                 ):

        self.root_dir = root_dir
        self.resolution = resolution
        self.max_val = max_val
        self.min_val = min_val
        self.image_channels = image_channels
        self.dataset_name = dataset_name
        if resolution in ms_training_res:
            ms_training_res.pop(ms_training_res.index(resolution))
        self.extra_num_img = len(ms_training_res)
        self.ms_training_res = sorted(ms_training_res, reverse=True)
        self.cfg_prob = cfg_prob
        self.serialize = serialize
        self.images_per_iter=images_per_iter

        if csv_file is None: # pylint: disable=R1720
            # loading annotations
            raise NotImplementedError("Only support csv for now")
            # self.items = read_txt(img_txt, serialize=serialize)
            # if self.items is not None:
            #     print(f'Reading {len(self.items)} img path')
            # else:
            #     print('No img file has been found')
            # self.captions = read_txt(text_txt, serialize=serialize)
            # if self.captions is not None:
            #     print(f'Reading {len(self.captions)} captions')
            #     assert(len(self.items) == len(self.captions))
            #     self.num_samples = len(self.captions)
            # else:
            #     print('No caption file has been found')
            #     self.num_samples = 0
        else:
            # print('loading from csv')
            if os.path.isdir(csv_file):
                csv_file = ','.join([os.path.join(csv_file, ff) for ff in os.listdir(csv_file) if ff.endswith('.csv')])
            else:
                assert os.path.isfile(csv_file)
            self.items, self.filepaths, self.captions = read_csv(csv_file, serialize=serialize)
            print(f'Build dataset {self.dataset_name}, contains{len(self.items)} samples in total.')
            self.num_samples = len(self.captions)
            
            self.fake_length = fake_length
            self.debug = debug
            self.max_samples = int(max_samples)
            if self.max_samples > 0:
                self.num_samples = min(self.num_samples, self.max_samples)

    def __del__(self):
        pass

    def loading_annotations(self):
        items = pd.read_csv(self.anno_name, encoding='utf-8')
        self.items = items.iloc[:, 0].tolist()
        self.captions = items.iloc[:, 1].tolist()
        self.num_samples = len(self.items)

    def get_text_only(self):
        caption = self.captions[int(np.random.randint(0, self.num_samples, 1))]
        return caption

    def get_text_subset(self, num=10000):
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        all_text = []
        for idx in indices[:num]:
            text = self.captions[int(idx)]
            all_text.append(text)
        return all_text

    def fetch_file(self, root_dir, filename):
        """Shortcut to reader's `fetch_file()`."""
        with open(os.path.join(root_dir, filename), 'rb') as f:
            return Image.open(f).convert('RGB')

    def crop_img(self, img):
        width, height = img.size
        shorted_side = min(width, height)
        if shorted_side == width:
            left = 0
            right = shorted_side
            top = (height - shorted_side) // 2
            bottom = (height + shorted_side) // 2
        else:
            left = (width - shorted_side) // 2
            right = (width + shorted_side) // 2
            top = 0
            bottom = shorted_side
        img = img.crop((left, top, right, bottom))
        return img

    def resize_img(self, img, size):
        img = img.resize((size, size), Image.LANCZOS)
        return img

    def get_raw_data(self, idx):
        image_path = self.filepaths[idx]

        if self.cfg_prob > 0. and np.random.rand() < self.cfg_prob:
            caption = 'XXX'
        else:
            caption = self.captions[idx]
            if not caption:
                caption = 'XXX'
            if self.dataset_name == 'JourneyDB' and np.random.rand() > 0.1:
                caption = '[Artistic] ' + caption
            elif self.dataset_name == 'SAM1B' and np.random.rand() > 0.1:
                caption = caption.removeprefix("The image features")
                caption = caption.removeprefix("The image depicts")
                if np.random.rand() > 0.3:
                    caption = caption.split("The style of")[0]

        # get caption
        #random_caption = self.get_text_only()

        # get image
        raw_img = self.fetch_file(self.root_dir, image_path)
        center_img = self.crop_img(raw_img)
        image = self.resize_img(center_img, self.resolution)
        image = np.array(image).astype(np.float32) / 255 * 2.0 - 1.0
        image = image.transpose(2, 0, 1)

        return_list = [image, caption]
        for extra_res in self.ms_training_res:
            res_i = int(extra_res)
            smaller_img_i = self.resize_img(center_img, res_i)
            smaller_img_i = np.array(smaller_img_i).astype(np.float32) / 255 * (self.max_val - self.min_val) + self.min_val
            smaller_img_i = smaller_img_i.transpose(2, 0, 1)
            return_list.append(smaller_img_i)
        return return_list

    @property
    def num_raw_outputs(self):
        return 4 + self.extra_num_img #[image, caption, random_caption, ...]

    @property
    def output_keys(self):
        keys = [f'image', 'text']
        for extra_res in self.ms_training_res:
            keys.append(f'image_{extra_res}')
        return keys

    def __getitem__(self, idx):
        n, raw_data_list, idx_ = 0, [], idx
        while True:
            try:
                raw_data = self.get_raw_data(idx_)
                raw_data_list.append(raw_data)
                n, idx_ = n+1, random.randint(0, self.num_samples)
            except Exception as e:
                print(f'loading {idx_} item fails, error details: {str(e)}')
                idx_ = random.randint(0, self.num_samples)
                raw_data = None
            if n >= self.images_per_iter: # images_per_iter can be 0
                break
            # if self.images_per_iter == n:
            #     break
            
        full_image = np.stack([dd[0] for dd in raw_data_list], axis = 0)
        full_text = [dd[1] for dd in raw_data_list]
        # full_text = np.stack([dd[1] for dd in raw_data_list], axis = 0)
        full_data = [full_image, full_text]
        assert isinstance(full_data, (list, tuple))
        assert len(full_data) == len(self.output_keys), 'Wrong keys!'
        return dict(zip(self.output_keys, full_data))

    def info(self):
        """Collects the information of the dataset.
        """
        dataset_info = {
            'Type': self.name,
            'Root dir': self.root_dir,
            'Num samples': self.num_samples,
            'Resolution': self.resolution,
            'dataset_name': self.dataset_name,
            'max_val': self.max_val,
            'min_val': self.min_val,
            'ms_training_res': self.ms_training_res,
            'extra_num_img': self.extra_num_img,
            'output_keys': self.output_keys,
        }
        return dataset_info

    def save_items(self, save_dir, tag=None):
        pass

