import os
import s3fs

import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint


# from s3_io import load_s3_image
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random


class ObjaverseS3Dataset(Dataset):
    """
    A dataset to prepare the objaverse images and their prompts
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        data_root,
        # tokenizer,
        s3_dirs_filename,
        captions_filename='filtered_cap3d.csv',
        view_type='offset0',
        size=1024,
        prompt_suffix="",
        prompt_prefix="",
        num_training=-1
    ):
        self.size = size
        self.prompt_suffix = prompt_suffix
        self.prompt_prefix = prompt_prefix
        # self.tokenizer = tokenizer

        # s3 = s3fs.S3FileSystem(anon=True)

        # data_root = 's3://phidias/rendered_images_v0.7/000-150/010e1d9c97474861a2366bc2cbf094e8/jiahao/renderings/00000010_rgb.png'
        # data_root = '/sensei-fs/users/jiahaoli/data/rendered_images_v0.7/'

        print('Loading image dirs on s3...')
        self.paths = [f'{data_root}/{x.strip()}' for x in open(s3_dirs_filename)]
        if num_training != -1:
            self.paths = self.paths[num_training:]
        print('Loaded image dirs on s3.')
        print('jiahao debug', len(self.paths))

        caption_path = captions_filename
        print(f'Loading captions from {caption_path}...')
        caption_dict = {}
        for line in open(caption_path):
            line = line.strip().split(',')
            caption_dict[line[0]] = ','.join(line[1:])
        print('Loaded captions.')

        print('Filtering valid paths...')
        self.paths = [x for x in self.paths if self.get_objaverse_id_from_s3_path(x) in caption_dict]
        self.captions = [caption_dict[self.get_objaverse_id_from_s3_path(x)] for x in self.paths]
        print('Filtering done.')

        print(f'Number of objects: {len(self.paths)}')

        self.image_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

        self.view_type = view_type
        assert self.view_type in ['random', 'offset0']
        self.num_views = 16
        assert self.num_views == 16

    def get_objaverse_id_from_s3_path(self, path):
        path = path.strip('/').split('/')
        # assert path[-1] == 'renderings' and path[-2] == 'uniform'
        return path[-1]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        example = {}

        index = index % len(self.paths)
        if self.view_type == 'random':
            starting_view_idx = float(np.random.randint(0, self.num_views))
            view_indices = [int(starting_view_idx + self.num_views // 4 * i) % self.num_views for i in range(4)]
        elif self.view_type == 'offset0':
            view_indices = [int(self.num_views // 4 * i) % self.num_views for i in range(4)]
        else:
            raise NotImplementedError
        image_filenames = [f'{view_idx:08d}_rgba.png' for view_idx in view_indices]
        instance_image_paths = [os.path.join(self.paths[index], 'uniform/renderings', name) for name in image_filenames]
        # s3 = s3fs.S3FileSystem(anon=True)
        # instance_images = [Image.open(self.s3.open(path, 'rb')) for path in instance_image_paths]
        # instance_images = [load_s3_i#mage(path) for path in instance_image_paths]
        instance_images = []
        for path in instance_image_paths:
            image = Image.open(path)
            image.load()

            background = Image.new("RGB", image.size, (255, 255, 255))
            mask_ = image.split()[3]
            background.paste(image, mask=image.split()[3])
            image = background
            instance_images.append(image)
        for image in instance_images:
            if not image.mode == "RGB":
                assert False
        # tile
        instance_images = [np.array(image) for image in instance_images]
        instance_image = np.concatenate([np.concatenate(instance_images[:2], axis=1),
                                         np.concatenate(instance_images[2:], axis=1)], axis=0)
        assert instance_image.shape[-1] == 3
        instance_image = Image.fromarray(instance_image)
        # resize
        instance_image = instance_image.resize((self.size, self.size), Image.LANCZOS)
        assert instance_image.height == instance_image.width == self.size
        # add to return dict
        example["jpg"] = self.image_transforms(instance_image)

        # read prompt
        instance_prompt = self.captions[index]
        if self.prompt_prefix:
            instance_prompt = self.prompt_prefix + ' ' + instance_prompt
        if self.prompt_suffix:
            instance_prompt = instance_prompt + ' ' + self.prompt_suffix
        # instance_prompt = instance_prompt + ', ' + self.prompt_suffix
        if False: # DEBUG: jiahao debug
            # if torch.distributed.get_rank() == 0: # DEBUG: jiahao debug
            import random
            vis_name = str(random.randint(0, 10000000))
            instance_image.save(f'./temp/vis/{vis_name}.png')
            with open(f'./temp/vis/{vis_name}.txt', 'w') as f:
                f.write(instance_prompt)
            # print(f'jiahao debug prompt for rank {torch.distributed.get_rank()}', instance_prompt) # DEBUG: jiahao debug
            print(f'jiahao debug prompt', instance_prompt) # DEBUG: jiahao debug
        example["txt"] = instance_prompt
        # example["txt"] = self.tokenizer(
        #     instance_prompt, # only get the first prompt
        #     padding="do_not_pad",
        #     truncation=True,
        #     max_length=self.tokenizer.model_max_length,
        # ).input_ids

        example['original_size_as_tuple'] = torch.tensor([instance_image.width, instance_image.height]).float()
        example['target_size_as_tuple'] = example['original_size_as_tuple'].clone()
        example['crop_coords_top_left'] = torch.tensor([0, 0]).float()

        return example


# def collate_fn(examples):
#     batch = {}
#     for key in examples[0].keys():
#         if isinstance(examples[0][key], torch.Tensor):
#             batch[key] = torch.stack([example[key] for example in examples], dim=0)
#         elif isinstance(examples[0][key], str):
#             batch[key] = [example[key] for example in examples]
#         else:
#             raise NotImplementedError
#     return batch


# def get_dataloader_from_config(config):
#     dataset = ObjaverseS3Dataset(**config.dataset_config)
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=config.batch_size,
#         shuffle=True,
#         collate_fn=collate_fn,
#         num_workers=config.num_workers,
#         # num_workers=0,
#     )
#     return dataloader
