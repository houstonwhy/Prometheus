import sys
import os
import json
import pickle as pkl
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import multiprocessing

try:
    from pcache_fileio import fileio
    pcache_root = 'pcache://vilabpcacheproxyi-pool.cz50c.alipay.com:39999/mnt/antsys-vilab_datasets_pcache_datasets'
    metadata_path = '/input/yyb/PlatonicGen/data/MVImgNet/metadata/mvimgnet_full_nocaption.pkl'
    data_root = os.path.join(pcache_root, 'yyb', 'MVImgNet')
except:
    print('pcache_fileio not install, work on local env')
    metadata_path = 'data/MVImgNet/local_tiny/metadata/mvimgnet_full_nocaption.pkl'
    data_root = '/data1/yyb/datasets/MVImgNet'

new_metadata_path = metadata_path.replace('.pkl', '_new.pkl')

class MetadataDataset(Dataset):
    def __init__(self, metadata, data_root):
        self.metadata = metadata
        self.data_root = data_root

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        kk, scene_meta = list(self.metadata.items())[idx]
        camera_json = os.path.join(self.data_root, scene_meta['camerasjson'])
        try:
            with open(camera_json, 'r') as jsf:
                cams = json.load(jsf)
            scene_meta['cameras'] = cams
        except:
            print(f'{camera_json} not exist')
            scene_meta['cameras'] = None
        return kk, scene_meta

def collate_fn(batch):
    new_metadata = {}
    for kk, scene_meta in batch:
        new_metadata[kk] = scene_meta
    return new_metadata

if __name__ == '__main__':
    with open(metadata_path, 'rb') as fp:
        metadata = pkl.load(fp)

    dataset = MetadataDataset(metadata, data_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=multiprocessing.cpu_count(), collate_fn=collate_fn)

    new_metadata = {}
    for batch in tqdm(dataloader):
        new_metadata.update(batch)


    with open(new_metadata_path, 'wb') as fp:
        pkl.dump(new_metadata, fp)