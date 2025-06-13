"""
Author: yyyyb
Last modified: 2024-09-10
Build metadata of MVImgNet for prometheus/director3d training
"""

import os
import os.path as osp
import random
import tyro
from tqdm import tqdm
import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader


# Set cwd to current file dir instead of workspace folder
current_file_path = osp.abspath(__file__)
current_directory = osp.dirname(current_file_path)
os.chdir(current_directory)

def fetch_txt(path):
    with open(path, 'r') as f:
        file_bytes = f.read()
    return file_bytes
class MVImgNetDataset(Dataset):
    def __init__(self, root_dir, category='all', use_caption=True, tiny = False):
        self.root_dir = root_dir
        self.tiny = tiny
        self.use_caption = use_caption
        self.scene_list = self.load_or_parse_scene_list(root_dir, category)

    def load_or_parse_scene_list(self, dataset_root, category):
        scene_list_txt = f'mvimgnet_scenes_{category}.txt'
        if osp.isfile(scene_list_txt) and not self.tiny:
            with open(scene_list_txt, 'r') as tx:
                scene_list = tx.read().split('\n')
                scene_list = [item.strip("/") for item in scene_list]
        else:
            scene_list = self.parse_scene_list(dataset_root, category)
        return scene_list

    def parse_scene_list(self, dataset_root, category='full'):
        scene_list = []
        if category == 'all':
            categories = os.listdir(dataset_root)
        else:
            categories = [category]

        for category in categories:
            if not osp.isdir(osp.join(dataset_root, category)):
                continue
            for idx in os.listdir(osp.join(dataset_root, category)):
                if not osp.isdir(osp.join(dataset_root, category, idx)):
                    continue
                scene_list.append(osp.join(dataset_root, category, idx))
        return scene_list

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        item = self.scene_list[idx]
        try:
            filenames = [osp.join(item,'images', f) for f in sorted(os.listdir(osp.join(self.root_dir, item, 'images'))) if f.endswith('jpg')]
            if len(filenames) < 8:
                return None
            if not osp.exists(osp.join(self.root_dir, item, 'cameras.json')):
                return None
            if len(filenames) != len(eval(fetch_txt(osp.join(self.root_dir, item, 'cameras.json')))):
                return None
            if self.use_caption:
                if not osp.exists(osp.join(self.root_dir, item, 'captions.txt')):
                    return None
                with open(osp.join(self.root_dir, item, 'captions.txt'), 'r', encoding='utf-8') as f:
                    #import ipdb; ipdb.set_trace()
                    rows = f.read().splitlines()
                captions = [row.replace('\n', '') for row in rows]
            else:
                captions = []
            #import ipdb; ipdb.set_trace()
            scene_meta = dict(
                name=item,
                filenames = filenames,
                category=item.split('/')[0],
                captions=captions,
                camerasjson=osp.join(item, 'cameras.json')
            )
            
            return scene_meta
        except Exception as e:
            print(Exception)
            print(f"Warning: {item} can not open, skip it")
            return None

def build_mvimgnet_metadata(
        root_dir: str = '/data1/yyb/datasets/MVImgNet',
        #root_dir: str = 'oss://antsys-vilab/datasets/pcache_datasets/yyb/MVImgNet',
        output_dir: str = '/data1/yyb/PlatonicGen/data/MVImgNet/local_tiny',
        use_caption: bool = False,
        category: str = 'all',
        mode: str = 'train',
        seed: int = 0,
        num_workers: int = 16,
        batch_size: int = 32,
        val_ratio: float = 0.15,
        tag: str = '_new',
):
    """Build metadata for mvimgnet dataset"""

    if mode == 'full':
        if 'oss:' in root_dir:
            try:
                from pcache_fileio import fileio # pylint: disable=E0401
                oss_folder_path = 'oss://antsys-vilab/datasets/pcache_datasets/'
                pcache_folder_path = 'pcache://vilabpcacheproxyi-pool.cz50c.alipay.com:39999/mnt/antsys-vilab_datasets_pcache_datasets/'
                root_dir = root_dir.replace(oss_folder_path, pcache_folder_path)
            except Exception:
                raise ValueError("pcacheio not install, work local workstation?")
        if not use_caption:
            mode += '_nocaption'
        dataset = MVImgNetDataset(root_dir, category, use_caption, tiny = ('tiny' in output_dir))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x)

        metadata = {}
        for batch in tqdm(dataloader):
            for scene_meta in batch:
                if scene_meta is not None:
                    metadata[scene_meta['name']] = scene_meta

        metadata_path = osp.join(output_dir, 'metadata', f'mvimgnet_{mode}.pkl')
        os.makedirs(osp.join(output_dir, 'metadata'), exist_ok=True)
        with open(metadata_path, 'wb') as fp:
            pkl.dump(metadata, fp)
        print(f'metadata save to {metadata}, contains {len(metadata)} items')
    elif mode == 'train':
        random.seed(seed)
        mode = 'full'
        if not use_caption:
            mode += '_nocaption'
        if tag:
            mode += tag
        try:
            full_metadata_path = osp.join(root_dir, 'metadata', f'mvimgnet_{mode}.pkl')
            assert osp.exists(full_metadata_path), "You need to create the full metadata first"
        except:
            full_metadata_path = osp.join(output_dir, 'metadata', f'mvimgnet_{mode}.pkl')
            assert osp.exists(full_metadata_path), "You need to create the full metadata first"
        with open(full_metadata_path, 'rb') as fp:
            full_metadata = pkl.load(fp)
        #print('??')
        val_samples = random.sample(list(full_metadata.keys()), round(len(full_metadata) * val_ratio))
        metadata_val = {k: full_metadata[k] for k in tqdm(val_samples)}
        metadata_train = {k: full_metadata[k] for k in tqdm(full_metadata) if k not in val_samples}

        if not output_dir:
            output_dir = root_dir
        metadata_train_path = full_metadata_path.replace('full', 'train').replace(root_dir, output_dir)
        metadata_val_path = full_metadata_path.replace('full', 'val').replace(root_dir, output_dir)
        os.makedirs(osp.join(output_dir, 'metadata'), exist_ok=True)    
        with open(metadata_train_path, 'wb') as fp:
            pkl.dump(metadata_train, fp)
        with open(metadata_val_path, 'wb') as fp:
            pkl.dump(metadata_val, fp)
        # save full meta in output dir again
        new_full_ = metadata_val_path.replace('val', 'full')
        with open(new_full_, 'wb') as fp:
            pkl.dump(full_metadata, fp)
        print(metadata_val_path)
if __name__ == "__main__":
    tyro.cli(build_mvimgnet_metadata)
    # build_mvimgnet_metadata()