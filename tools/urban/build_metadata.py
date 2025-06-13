"""
Author: yyyyb
Last modified: 2024-08-30
Load metadata of UrbanGen can creat scene seqence for director 3d training
Create sequence list
"""
import os
import os.path as osp
import pickle as pkl
from pathlib import Path
import tyro
from easydict import EasyDict
import numpy as np





def convert_from_urbangen_metadata(
        root_dir: str = '/data1/yyb/datasets/UrbanGen/datasets4training/urban/metadata/urbangiraffe',
        output_dir: str = 'data/UrbanGen/local_tiny/',
        with_caption: bool = False,
        max_seq_length: str = 200,
        sequence_overlap: str = 120,
        mode: str = 'full',
):
    seq_max_overlap = {
        'kitti':(32,10),
        'kitti360':(32,10),
        'nuscenes':(24,8),
        'waymo':(48,16)
    }
    raw_meta_path = osp.join(root_dir, 'kitti-360_semantic-kitti_nuscenes_waymo_train.pkl')

    if not with_caption:
        mode += '_nocaption'
    with open(raw_meta_path, 'rb') as fp:
        raw_meta = pkl.load(fp)
    valid_keys = ['index', 'cam_K' ,'cam2world','image_path']
    dataset_bucket = EasyDict({
        'kitti360':{},
        'kitti':{},
        'nuscenes':{},
        'waymo':{},

    })
    # split into each dataset
    for frame in list(raw_meta):
        if 'kitti-360' in frame['image_path']:
            dataset_type = 'kitti360'
            seq_id = frame['image_path'].split('/')[3] 
            # 'datasets/kitti-360/data_2d_raw/2013_05_28_drive_0003_sync/image_00/data_rect/000000.png'
        elif 'semantic-kitti' in frame['image_path']:
            dataset_type = 'kitti'
            seq_id = frame['image_path'].split('/')[4]
        elif 'nuscenes' in frame['image_path']:
            dataset_type = 'nuscenes'
            seq_id = frame['image_path'].split('/')[3] 
        elif 'waymo' in frame['image_path']:
            dataset_type = 'waymo'
            seq_id = frame['voxel_path'].split('/')[3]

        if seq_id in dataset_bucket[dataset_type].keys():
            dataset_bucket[dataset_type][seq_id].append(frame)
        else:
            dataset_bucket[dataset_type][seq_id] = [frame]
    
    urbangen_seqs = {}
    # We Keep each sequence in 10s
    # Split KITTI/KITTI-360 Long sequences into small ones
   
    # for seq_id, long_seq in dataset_bucket['kitti360'].items():
    #     pass

    
    for dataset_name in ['kitti360', 'kitti', 'waymo', 'nuscenes']:
        sub_seqs = {}
        max_seq_length, sequence_overlap = seq_max_overlap[dataset_name]
        step = max_seq_length - sequence_overlap
        window_size = max_seq_length
        for long_seq_name, long_seq in dataset_bucket[dataset_name].items():
            seq_id = 0
            if dataset_name == 'kitti360':
                long_seq_name = long_seq_name.split('_')[4]
            for start in range(0, len(long_seq) - window_size + 1, step):
                end = start + window_size
                subsequence = [{k : ff[k] for k in valid_keys} for ff in long_seq[start:end]]
                poses = np.array([ff['cam2world'][:3,3] for ff in subsequence])
                dis = np.linalg.norm(np.max(poses, axis=0) - np.min(poses, axis=0), 2)
                if dis.max() < 120 and dis.max() > 10:
                    sub_seqs[f'{dataset_name}_{long_seq_name}_{seq_id}'] = subsequence
                    seq_id += 1
                else:
                    pass
                    #print(f'{dataset_name}_{long_seq_name}_{seq_id} has distance {dis}')
        print(f'{dataset_name} has {len(sub_seqs)} vallid seqs')
        urbangen_seqs.update(sub_seqs)

    # if not with_caption:
    #     for seq in urbangen_seqs.values():
    #         for ff in seq:
    #             ff['captions'] = ''
                
    output_path= Path(osp.join(output_dir, 'metadata', f'urbangen_{mode}.pkl'))
    if not output_path.parent.is_dir():
        os.makedirs(output_path.parent)
    with open(output_path, 'wb') as fp:
         pkl.dump(urbangen_seqs, fp)

    print('Done')


if __name__ == "__main__":
    tyro.cli(convert_from_urbangen_metadata)