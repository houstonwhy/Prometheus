"""
Contains the class of Base and Joint dataset classes.
"""
#pylint: disable=import-error
import os
import cv2
import datetime
import random
import io
import pickle as pkl
import json
from PIL import Image
from abc import ABC, abstractmethod
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
# import time
from prometheus.utils import import_str
from prometheus.utils.misc import parse_file_format
from prometheus.datasets.file_readers import build_file_reader
from prometheus.datasets.transformations import build_transformation
from prometheus.datasets.transformations.misc import switch_between
from prometheus.datasets.utils import _check_valid_rotation_matrix,  matrix_to_square
from prometheus.datasets.transformations.utils.formatting_utils import format_image
# from concurrent.futures import ThreadPoolExecutor, TimeoutError

__all__ = ['JointDataset', 'ProbDataset', 'BaseDataset']

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timeout=seconds)
                except TimeoutError:
                    future.cancel()
                    raise TimeoutError(f'Timeout: took more than {seconds} seconds. Returning None.')
                    # return None
                return result
        return wrapper
    return decorator


class JointDataset(Dataset):
    #TODO
    """
    JointDataset that contains multiplt different datasets that may has different format of output -> each batch will concat result togather 
    To avoid different dataset has different length, it will:
    1. if fake length was assigned -> 
    2. if fake length was disable, the lenght of a joint dataset is determained by the longgent
    """
    def __init__(self,
                 datasets,
                 fake_length : int =-1,
                 debug : bool = False
                 ):
        self.datasets = []
        self.fake_length = fake_length
        self.debug = debug
        # 1. Init multiple datasets
        for dataset in datasets:
            self.datasets.append(import_str(dataset['module'])(**dataset['args'], fake_length=fake_length, debug = debug)) 
            # disable fake lenght of subdatasets to avoid random sampling for two times.
        self.dataset_leght_max = max(len(d) for d in self.datasets)

    def __len__(self):
        return self.dataset_leght_max if self.fake_length == -1 else min(self.fake_length, self.dataset_leght_max)

    def __getitem__(self, index):
        out = {}
        for dataset in self.datasets:
            if self.fake_length != -1: # if use fake lenght, random sample
                idx_ = np.random.randint(0, len(dataset))
            else:
                idx_ = index % len(dataset)
            # try:
            tt = dataset.__getitem__(idx_)
            out.update(tt)
            # except:
            #     print(('??'))

        return out

class ProbDataset(Dataset):
    """ProbDataset that contains multiplt datasets with the same format of output
    if probs was assigned: it will ignore the reallength of each datasets, and sample on a condirion, the length of each dataset is manully given by a fake length
    """
    def __init__(self,
                 datasets,
                 probs: list = None,
                 fake_length: int = -1,
                 debug: bool = False
                 ):

        #super().__init__(fake_length, debug)
        self.debug = debug
        self.fake_length=fake_length
        self.probs = probs
        self.datasets = []

        for dataset in datasets:
            self.datasets.append(import_str(dataset['module'])(**dataset['args'], fake_length=-1, debug = debug)) # disable fake lenght of subdatasets to avoid random sampling for two times.

        # set prob and init index for each datasets
        if not probs:
            probs = [len(dd) for dd in self.datasets]
        # normalize
        self.probs = [pp / sum(probs) for pp in probs]
        print('Datasets with probs')
        assert len(self.probs) == len(self.datasets)
        self.ids = [0 for _ in self.datasets]
        # assert sum(self.probs) == 1
        print(f'{sum(self.probs)}')
        self.dataset_lenght_sum = sum(len(d)for d in self.datasets)
        print("Dataset\tProbability")
        for dataset, probability in zip(self.datasets, self.probs):
            print(f"{dataset.dataset_name}\t{probability:.2f}")

    def __len__(self):
        return self.dataset_lenght_sum if self.fake_length == -1 else min(self.fake_length, self.dataset_lenght_sum)
    

    def __getitem__(self, index):
        #print(index)
        while True:
            di, dataset = random.choices(list(enumerate(self.datasets)), weights=self.probs, k=1)[0]
            if self.fake_length != -1:
                #idx_ = random.randint(0, len(dataset))
                idx_ = np.random.randint(0, len(dataset))
            else:
                idx_ = self.ids[di] % len(dataset)
                self.ids[di] += 1
                self.ids[di] = self.ids[di] % len(dataset)
            
            try:
                data_item = dataset.__getitem__(idx_)
                if data_item is not None:
                    break
            except Exception as e:
                    process_id = os.getpid()
                    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f'[{current_time}] PID: {process_id} - loading item from {dataset.dataset_name} fails, error details: {str(e)}')
                    data_item = None
                #  idx_ = int(np.random.randint(0, self.num_samples, 1))
                # if n > self.max_retries:
                #     raise ValueError("Failed to load after max retries")

        # print(f'[{current_time}] PID: {process_id} - loading item{idx_}  from {self.dataset_name} fails, error details: {str(e)}')

        return data_item

class BaseDataset(Dataset):
    """XX"""
    def __init__(self,
                 root_dir,
                 file_format=None,
                 annotation_path = None,
                 annotation_meta=None,
                 annotation_format='json',
                 dataset_name=None,
                 max_samples=-1,
                 mirror=False,
                 img_size = 256,
                 transform_kwargs=None,
                 fake_length=-1,
                 max_retries=10,
                 debug = False
                 ):
        super().__init__()
        self.debug = debug
        self.max_retries = max_retries
        self.fake_length = fake_length
        # Build file reader to read data from disk.
        self.root_dir = root_dir
        if not dataset_name:
            self.dataset_name = os.path.splitext(os.path.basename(self.root_dir))[0]
        else:
            self.dataset_name = dataset_name
        #import ipdb; ipdb.set_trace()
        if file_format is None:
            file_format = parse_file_format(root_dir)
        assert file_format is not None, 'Unparsable file format from root dir!'
        self.file_format = file_format.lower()
        self.reader = build_file_reader(self.file_format)

        # # Load/build metadat
         # Parse item list of the dataset.
        self.annotation_path = annotation_path
        self.annotation_meta = annotation_meta
        if isinstance(annotation_format, str):
            annotation_format = annotation_format.lower()
        self.items, self.metadata = None, None

        if annotation_path and os.path.isfile(annotation_path):
            if annotation_format is None:
                annotation_format = parse_file_format(annotation_path)
            if annotation_format == 'pkl':
                fp = open(annotation_path, 'rb')
            elif annotation_format == 'npz':
                fp = annotation_path
            else:
                fp = open(annotation_path, 'r')
        # # First option: use `annotation_path` if available.
        # if annotation_path and os.path.isfile(annotation_path):
        #     # File is closed after parsed by `self.parse_annotation_file()`.
        #     fp = open(annotation_path, 'r')  # pylint: disable=consider-using-with
        # Second option: use `annotation_meta` if available.
        elif annotation_meta:
            if annotation_format is None:
                annotation_format = parse_file_format(annotation_meta)
            fp = self.reader.open_anno_file(root_dir, annotation_meta)
        # No external annotation is provided.
        else:
            fp = None

        self.annotation_format = annotation_format
        if fp is not None and fp :  # Use external annotation if available.
            self.metadata = self.parse_annotation_file(fp)
            if not isinstance(fp, str):
                fp.close()
        else:  # Fallback: use image list provided by `self.reader`.
            #self.items = self.reader.get_image_list(root_dir)
            self.metadata = self.build_metadata()
        self.items = list(self.metadata.keys())

        assert isinstance(self.items, list) and len(self.items) > 0
        self.dataset_samples = len(self.items)
        self.num_samples = self.dataset_samples
        # Cut off the dataset if needed.
        self.max_samples = int(max_samples)
        if self.max_samples > 0:
            self.num_samples = min(self.num_samples, self.max_samples)
            
        # Mirror aug & data transform pipeline
        self.img_size = img_size
        self.mirror = bool(mirror)
        if self.mirror:
            self.num_samples = self.num_samples * 2
        self.transforms = dict()
        self.transform_kwargs = transform_kwargs or dict()
        self.parse_transform_config()
        self.build_transformations()
        print(f'Build dataset {dataset_name}, contains{self.num_samples} samples in total.')

    # @abstractmethod
    def build_metadata(self):
        """"""
        raise NotImplementedError('Should be implemented in derived class!')

    def parse_annotation_file(self, fp):
        """Parses items according to the given annotation file.

        The base class provides a commonly used parsing method, which is to
        parse a JSON file directly, OR parse a TXT file by treating each line as
        a `space-joined` string.

        Please override this function in derived class if needed.

        Args:
            fp: A file pointer, pointing to the opened annotation file.

        Returns:
            A parsed item list.

        Raises:
            NotImplementedError: If `self.annotation_format` is not implemented.
        """
        if self.annotation_format == 'json':
            return json.load(fp)

        if self.annotation_format == 'pkl':
            return pkl.load(fp)

        if self.annotation_format == 'npz':
            return np.load(fp, allow_pickle=True)

        if self.annotation_format == 'txt':
            items = []
            for line in fp:
                fields = line.rstrip().split(' ')
                if len(fields) == 1:
                    items.append(fields[0])
                else:
                    items.append(fields)
            return items

        raise NotImplementedError(f'Not implemented annotation format '
                                  f'`{self.annotation_format}`!')

    def __del__(self):
        """Destroys the dataset, particularly closes the file reader."""
        if hasattr(self, 'reader'):
            self.reader.close(self.root_dir)

    @property
    def name(self):
        """Returns the class name of the dataset."""
        return self.__class__.__name__

    @property
    def num_raw_outputs(self):
        """Returns the number of raw outputs.

        This function should align with `self.get_raw_data()`, and is
        particularly used by `datasets/data_loaders/dali_pipeline.py`
        """
        raise NotImplementedError('Should be implemented in derived class!')

    @property
    def output_keys(self):
        """Returns the name of each output within a pre-processed item.

        This function should align with `self.transform()`, and is particularly
        used by `self.__getitem__()` as well as
        `datasets/data_loaders/dali_batch_iterator.py`.
        """
        raise NotImplementedError('Should be implemented in derived class!')

    def __len__(self):
        """Gets the total number of samples in the dataset."""
        return self.num_samples if self.fake_length == -1 else min(self.fake_length, self.num_samples)

    def fetch_file(self, filename):
        """Shortcut to reader's `fetch_file()`."""
        return self.reader.fetch_file(self.root_dir, filename)
        
    def fetch_img(self, root_dir, filename):
        """Shortcut to reader's `fetch_file()`."""
        with open(os.path.join(root_dir, filename), 'rb') as f:
            with Image.open(f) as img:  # 使用with确保文件句柄正确释放
                return img.convert('RGB')
            # return Image.open(f).convert('RGB')

    def save_items(self, save_dir, tag=None):
        """Saves the item list to disk.

        Name of the saved file is set as `${self.dataset_name}_item_list.txt`

        Args:
            save_dir: The directory under which to save the item list.
        """
        if tag is None:
            save_name = f'{self.dataset_name}_item_list.txt'
        else:
            assert isinstance(tag, str), 'String is required!'
            save_name = f'{self.dataset_name}_{tag}_item_list.txt'
        save_path = os.path.join(save_dir, save_name)

        os.makedirs(save_dir, exist_ok=True)
        # TODO: Check file existence and handle such a case?
        with open(save_path, 'w') as f:
            for item in self.items:
                if isinstance(item, (list, tuple)):
                    item_str = ' '.join(map(str, item))
                else:
                    item_str = str(item)
                f.write(f'{item_str}\n')

    def info(self):
        """Collects the information of the dataset.

        Please append new information in derived class if needed.
        """
        dataset_info = {
            'Type': self.name,
            'Root dir': self.root_dir,
            'Dataset name': self.dataset_name,
            'Dataset file format': self.file_format,
            'Annotation path': self.annotation_path,
            'Annotation meta': self.annotation_meta,
            'Annotation format': self.annotation_format,
            'Num samples in dataset': self.dataset_samples,
            'Num samples to use (non-positive means all)': self.max_samples,
        }
        return dataset_info

    def get_raw_data(self, idx):
        """Gets raw data of a particular item.

        Args:
            idx: Index of the item within the item list maintained by the
                dataset.

        Returns:
            The raw data fetched by the file reader.
        """
        raise NotImplementedError('Should be implemented in derived class!')

    def parse_transform_config(self):
        image_channels = self.transform_kwargs.setdefault('image_channels', 3)
        min_val = self.transform_kwargs.setdefault('min_val', -1.0)
        max_val = self.transform_kwargs.setdefault('max_val', 1.0)
        use_square = self.transform_kwargs.setdefault('use_square', False)
        center_crop = self.transform_kwargs.setdefault('center_crop', False)
        image_size = self.transform_kwargs.get('image_size', self.img_size)
        aspect_ratio =  self.transform_kwargs.get('aspect_ratio', (1,1))
        if not use_square:
            image_size = (int(aspect_ratio[0] * image_size), int(aspect_ratio[1] * image_size))
            
        self.transform_config = dict(
            decode=dict(transform_type='Decode', image_channels=image_channels,
                        return_square=use_square, center_crop=center_crop),
            resize=dict(transform_type='Resize', image_size=image_size),
            normalize=dict(transform_type='Normalize',
                           min_val=min_val, max_val=max_val)
        )


    def build_transformations(self):
        """Builds each individual transformation in the pipeline."""
        # Particularly used for data mirroring.
        self.transforms['_mirror'] = build_transformation(
            transform_type='Flip', horizontal_prob=1.0, vertical_prob=0.0)
        self.ToTensor = torchvision.transforms.ToTensor()
        # Build data transformations.
        for name, config in self.transform_config.items():
            self.transforms[name] = build_transformation(**config)

        # To enable DALI pre-processing, all the nodes within the
        # transformation pipeline should support DALI.
        self.support_dali = all(trans.support_dali
                                for trans in self.transforms.values())

        # Check if any transformation node is implemented with customized
        # function.
        self.has_customized_function_for_dali = any(
            trans.has_customized_function_for_dali
            for trans in self.transforms.values())
    
    def mirror_aug(self, data, do_mirror, use_dali=False):
        """Mirrors (i.e., horizontal flips) the data to double the dataset.

        Args:
            data: The data to mirror.
            do_mirror: Whether mirroring is needed.
            use_dali: Whether the input data is a node from DALI pre-processing
                pipeline. (default: False)
        """
        flipped_data = self.transforms['_mirror'](data, use_dali=use_dali)
        return switch_between(cond=do_mirror,
                              cond_true=flipped_data,
                              cond_false=data,
                              use_dali=use_dali)

    def sample_views(self, num_frames, sample_rate, num_input_views, num_novel_views):
        indices = np.arange(num_frames)
        if len(indices) > ((num_input_views-1) * sample_rate + 1):
            random_start = np.random.randint(0, len(indices) - ((num_input_views-1) * sample_rate + 1))
            sample_rate_ = sample_rate
        else:
            random_start = 0
            sample_rate_ = (num_frames + 1) // num_input_views
            print(f"Not enough image frame, need {(num_input_views-1) * sample_rate_ + 1}, only has {len(indices)} frames in seq, reduce sample_rate from {sample_rate} to {sample_rate_}")

        input_view_ids = indices[random_start:][::sample_rate_][:num_input_views]
        if len(input_view_ids) < num_input_views:
            delta = num_input_views - len(input_view_ids)
            indices_ = np.random.choice(indices[random_start:], delta, replace=delta>indices[random_start:])
            input_view_ids = np.concatenate((input_view_ids, indices_))
            input_view_ids.sort()
        # Novel views ids
        novel_view_ids = random.sample(indices[random_start:random_start+(num_input_views-1) * sample_rate_+1].tolist(), k=num_novel_views)

        novel_view_ids_ = np.concatenate((indices[random_start:max(input_view_ids)+1], input_view_ids)) 
        unique_values, counts = np.unique(novel_view_ids_, return_counts=True)
        novel_view_ids_ = unique_values[counts == 1]
        novel_view_ids = np.random.choice(novel_view_ids_ ,num_novel_views, replace=num_novel_views > len(novel_view_ids_))
        novel_view_ids.sort()
        full_ids = np.concatenate((input_view_ids, novel_view_ids))
        return full_ids

    def __getitem__(self, idx):
        idx_, n = idx, 0
        if self.debug: # get error
            while True:
                #raw_data = self.get_raw_data(idx_)
                raw_data = self.get_raw_data(idx_)
                if raw_data is not None:
                    break
                idx_ = int(np.random.randint(0, self.num_samples, 1))
                n += 1
                if n > self.max_retries:
                    return None
        else:
            while True:
                #raw_data = self.get_raw_data(idx_)
                try:
                    raw_data = self.get_raw_data(idx_)
                    if raw_data is not None:
                        break
                except Exception as e:
                    process_id = os.getpid()
                    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f'[{current_time}] PID: {process_id} - loading idx {idx_} item {self.items[idx_]} from {self.dataset_name} fails, error details: {str(e)}')
                    raw_data = None
                idx_ = int(np.random.randint(0, self.num_samples, 1))
                n += 1
                if n > self.max_retries:
                    raise ValueError(f"[{current_time}] PID: {process_id} - loading from {self.dataset_name}  Failed to load after max retries")

        # print(f'[{current_time}] PID: {process_id} - loading item{idx_}  from {self.dataset_name} fails, error details: {str(e)}')

        # raw_data = self.get_raw_data(idx)
        assert isinstance(raw_data, (list, tuple))
        assert len(raw_data) == len(self.output_keys), 'Wrong keys!'
        return dict(zip(self.output_keys, raw_data))

    def crop(self, image, crop_pos):
        height, width = image.shape[0:2]
        crop_size = min(height, width)
        y = int((height - crop_size) * crop_pos)
        x = int((width - crop_size) * crop_pos)
        image_croped = np.ascontiguousarray(image[y:y + crop_size, x:x + crop_size])
        return image_croped

    def get_image(self, image_path, crop_pos = 0.5, do_mirror=False):
        # buffer = np.frombuffer(self.fetch_file(image_path), dtype=np.uint8)
        # raw_image = format_image(cv2.imdecode(raw_image, cv2.IMREAD_UNCHANGED))
        raw_image = np.array(self.fetch_img(self.root_dir, image_path))
        image = self.crop(raw_image, crop_pos)
        image = self.transforms['resize'](image, use_dali=False)
        image = self.mirror_aug(image, do_mirror, use_dali=False)
        image = self.transforms['normalize'](image, use_dali=False)[:3]

        return raw_image, image


class MultiviewDataset(BaseDataset):
    def __init__(self,
                root_dir,
                file_format=None,
                annotation_path = None,
                annotation_meta=None,
                annotation_format='json',
                view_type='uniform',# 'uniform', 'random'
                dataset_name=None,
                max_samples=-1,
                mirror=False,
                img_size = 256,
                transform_kwargs=None,
                fake_length=-1,
                scene_scale_threshold=0.0,
                use_seq_aug=False,
                #use_pose_aug=False,
                debug = False
                ):
        self.view_type = view_type
        self.scene_scale_threshold = scene_scale_threshold
        self.use_seq_aug = use_seq_aug
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

    @property
    def num_raw_outputs(self):
        return 5  #[image, cameras, text]

    @property
    def output_keys(self):
        keys = ['images_mv', 'cameras_mv', 'text_mv', 'dataset_name_mv', 'scene_name_mv']
        # for extra_res in self.ms_training_res:
        #     keys.append(f'image_{extra_res}')
        return keys
    
    @abstractmethod
    def parse_scene(self, idx):
        raise NotImplementedError('Should be implemtent by specific dataset')
    
    @abstractmethod
    def get_frames_data(self):
         raise NotImplementedError('Should be implemtent by specific dataset')

    #TODO a unified multiview sample&aug function
    # @timeout(10)
    def process_cameras(self, c2ws, intrinsics, num_input_views = -1):
        if self.normalized_cameras:
            
            ref_w2c = torch.inverse(matrix_to_square(c2ws[:1]))
            c2ws = (ref_w2c.repeat(c2ws.shape[0], 1, 1) @ matrix_to_square(c2ws))[:,:3,:]
            
            nv = num_input_views if num_input_views > -1 else c2ws.shape[0] # calc scale on input views only or all views?
            T_norm = c2ws[:nv, :3, 3].norm(dim=-1).max()
            c2ws[:, :3, 3] = c2ws[:, :3, 3] / (T_norm + 1e-2)
            if T_norm < self.scene_scale_threshold:
                raise ValueError(f"Camera motion is too small, {T_norm} < {self.scene_scale_threshold}")

        if (not _check_valid_rotation_matrix(c2ws[:, :3, :3], tol=1e-5)) or ((c2ws[:, :3, 3].norm(dim=-1) > 5).any()):
            raise ValueError("Not a valid rotation matrix")

        cameras = torch.cat([c2ws.flatten(1, 2), intrinsics], dim=1).to(torch.float32)
        return cameras
    
    def get_caption(self, scene_data):
        if not self.use_caption or random.random() < self.drop_text_p:
            text = 'XXX'
        else:
            #text = 'scene_dataset_name'
            #TODO better caption parseing process
            if 'data1' in self.root_dir: # working on local workstation
                captions_root = '/data1/yyb/PlatonicGen/workdir/captions'
            elif ('pcache' in self.root_dir) or ('input' in self.root_dir) :
                captions_root = '/input/yyb/PlatonicGen/outputs/captions'
            else:
                raise ValueError(f'Unknown caption dir {captions_root}')
            
            caption_path = os.path.join(captions_root, scene_data['dataset_name'], scene_data['scene_name']+".txt")
            
            if os.path.exists(caption_path):
                with open(caption_path, 'r') as fp:
                    captions = fp.read().split('\n')
                text = random.choice(captions)
            elif self.use_caption and not self.debug:
                text = ''
                raise ValueError("No caption, skip")
            else:
                text = 'ZZZ'
            # if 'captions' in scene_data.keys() and len(scene_data['captions']) >= 1:
            #     text = random.choice(scene_data['captions'])
            # else:
            #     text = ''
        return text
    
    def sample_views(self, num_frames, sample_rate, num_input_views, num_novel_views, view_type = 'uniform'):
        """Sample imple/novel from given seq

        Args:
            num_frames (_type_): _description_
            sample_rate (_type_): _description_
            num_input_views (_type_): _description_
            num_novel_views (_type_): _description_
            view_type: novel view sample type ['uniform'. random]
        Returns:
            full ids
        """
        indices = np.arange(num_frames)

        if sample_rate == -1: # random sample view from full sequqnce
            #TODO add random view sample for Objaverse -> later apply to all datasets
            random_start = 0
            full_ids = np.random.choice(indices, num_input_views + num_novel_views, replace= False)

            max_idx, min_idx =  np.max(full_ids),np.min(full_ids)
            tt = full_ids[(full_ids != max_idx) & (full_ids != min_idx)]
            # try:
            oo = np.random.choice(tt, num_input_views - 2, replace= False)
            # except:
            #     import ipdb; ipdb.set_trace()
            input_view_ids = np.concatenate(([max_idx, min_idx], oo))
            novel_view_ids = full_ids[~np.isin(full_ids, input_view_ids)]
            
        elif sample_rate > 0:
            if len(indices) > ((num_input_views-1) * sample_rate + 1):
                random_start = np.random.randint(0, len(indices) - ((num_input_views-1) * sample_rate + 1))
                sample_rate_ = sample_rate
            else:
                random_start = 0
                sample_rate_ = (num_frames + 1) // num_input_views        
            if sample_rate_ < 1:
                # has no room to sample novel view and num_novel view > 0 ->return
                raise ValueError("Not enough images to sample input views, sample_rate < 1")

            input_view_ids = indices[random_start:][::sample_rate_][:num_input_views]
            if len(input_view_ids) < num_input_views:
                delta = num_input_views - len(input_view_ids)
                indices_ = np.random.choice(indices[random_start:], delta, replace=delta>indices[random_start:])
                input_view_ids = np.concatenate((input_view_ids, indices_))
                input_view_ids.sort()
            # Novel views ids
            if num_novel_views == 0:
                # return if do not need to sample novel view
                return input_view_ids

            novel_view_ids_ = np.concatenate((indices[random_start:max(input_view_ids)+1], input_view_ids)) 
            unique_values, counts = np.unique(novel_view_ids_, return_counts=True)
            novel_view_ids_ = unique_values[counts == 1]
            if len(novel_view_ids_) == 0 and num_novel_views > 0:
                raise ValueError(f"Not enough novel views, has {len(novel_view_ids_)} only")
            novel_view_ids = np.random.choice(novel_view_ids_ ,num_novel_views, replace=num_novel_views > len(novel_view_ids_))
        else:
            raise ValueError('Illegal samplea rate {sample_rate}')
        
        input_view_ids.sort(), novel_view_ids.sort()
        full_ids = np.concatenate((input_view_ids, novel_view_ids))
        return full_ids
    
    @timeout(1800)
    def get_raw_data(self, idx):
        # 1.Parse sequence  (dataset specific)
        scene_data = self.parse_scene(idx)

        dataset_name, scene_name = scene_data['dataset_name'], scene_data['scene_name']

        # 2.Get Caption (general)
        text = self.get_caption(scene_data=scene_data)
        # if self.use_caption and (text is None):
        if self.use_caption and (text == ''):
            raise ValueError("No caption, skip")

        # 3.Sample views
        full_ids = self.sample_views(
            num_frames=scene_data['num_frames'],
            sample_rate=scene_data['sample_rate'],
            num_input_views=scene_data['num_input_views'],
            num_novel_views=scene_data['num_novel_views']
        )
        # if full_ids is None:
        #     return None

        # 4. Get full data frame-by-frame (dataset specific)
        #TODO add more image / trajectory augmentation
        do_mirror, crop_pos, flip_traj = False, 0.5, False
        images, c2ws, intrinsics = self.get_frames_data(
            scene_data = scene_data, 
            indices=full_ids,
            do_mirror = do_mirror, 
            crop_pos = crop_pos, 
            flip_traj = flip_traj
            )
        
        # 5. Pose Normalization (general)
        cameras = self.process_cameras(c2ws=c2ws, intrinsics=intrinsics, num_input_views=scene_data['num_input_views'])
        # if cameras is None:
        #     return None

        return [images, cameras, text, dataset_name, scene_name]
    