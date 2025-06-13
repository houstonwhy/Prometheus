"""
Author: yyyyb
Last modified: 2024-08-16
Build metadata of DL3DV for prometheus/director3d training
"""
import os
import os.path as osp
from copy import deepcopy
import random
import pickle as pkl
import tyro

def build_dl3dv_metadata(
        root_dir: str = '/data1/yyb/datasets/DL3DV-10K',
        output_dir: str = '/data1/yyb/PlatonicGen/data/DL3DV-10K',
        # output_dir: str = '/data1/yyb/PlatonicGen/data/DL3DV-10K/local_tiny',
        mode: str = 'train',
        use_caption: bool = True,
        seed: int = 0,
        # use_depth: bool = True
        val_ratio: float = 0.15,
        max_seq_length: str = 64,
        sequence_overlap: str = 16,
):
    """Build metada for dl3dv dataset"""
    if mode == 'full':
        os.makedirs(osp.join(output_dir, 'metadata'), exist_ok= True)
        items, metadata = [], {}
        example_item = '0a3e9c8e9b7713e77f45bf55edd1190c925028293aa0561feec54c826a0e6b98'
        items = os.listdir(osp.join(root_dir))
        items = list(filter(lambda item: (len(item) == len(example_item)) and osp.isdir(osp.join(root_dir, item)), items))
        for item in items:
            filenames = [osp.join(item, f) for f in sorted(os.listdir(osp.join(root_dir, item, 'images_4'))) if f.endswith('png')]

            scene_meta = dict(
                name = item,
                filenames = filenames,
                #images = sorted(os.listdir(osp.join(basedir, 'images_4'))),
                transforms_json = osp.join(item, 'transforms.json')
                )
            metadata[item] = scene_meta
        items = list(metadata.keys())

        # further filter by if has captions
        if use_caption:
            for item in items:
                if osp.exists(osp.join(root_dir, item, 'captions.txt')):
                    # .captions[basedir] = []
                    with open(osp.join(root_dir, item, 'captions.txt'), 'r', encoding='utf-8')as f:
                        rows = f.readlines()
                    #captions[basedir] = [row.replace('\n', '') for row in rows]
                    metadata[item]['captions'] = [row.replace('\n', '') for row in rows]
            metadata = {k :v for k, v in metadata.items() if 'captions' in v.keys()}
        # else:
        #     mode += '_nocaption'
        metadata_path =  osp.join(output_dir, 'metadata', f'dl3dv_{mode}.pkl')
        with open(metadata_path, 'wb') as fp:
            pkl.dump(metadata, fp)

    elif mode == 'train':
        random.seed(seed)
        mode = 'full'
        if not use_caption: # for local 
            mode += '_nocaption'
        try:
            full_metadata_path = osp.join(root_dir, 'metadata', f'dl3dv_{mode}.pkl')
            assert osp.exists(full_metadata_path), "You need to create the full metadata first"
        except:
            full_metadata_path = osp.join(output_dir, 'metadata', f'dl3dv_{mode}.pkl')
            assert osp.exists(full_metadata_path), "You need to create the full metadata first"
        with open(full_metadata_path, 'rb') as fp:
            full_metadata_raw = pkl.load(fp)
        if len(list(full_metadata_raw.keys())[0].split('_')) > 1: # mini clip already
            sub_seqs = full_metadata_raw
        else:
        # Create mini clip
            sub_seqs = {}
            step = max_seq_length - sequence_overlap
            window_size = max_seq_length
            for long_seq_name, long_seq in full_metadata_raw.items():
                seq_id = 0
                for start in range(0, len(long_seq['filenames']) - window_size + 1, step):
                    end = start + window_size
                    subsequence = deepcopy(long_seq)
                    subsequence['filenames']=  long_seq['filenames'][start:end]
                    #sub_seqs[f'{long_seq_name}_{seq_id}'] = subsequence
                    if len(subsequence['filenames']) > (max_seq_length / 2):
                        sub_seqs[f'{long_seq_name}_{seq_id}'] = subsequence
                        seq_id += 1
                    # else:
                    #     pass
            print(f'Full DL3DV has {len(sub_seqs)} vallid seqs')
            # Create sub sequqnce

            with open(full_metadata_path.replace('.pkl', '_miniclip.pkl'), 'wb') as fp:
                pkl.dump(sub_seqs, fp)

        val_samples = random.sample(list(sub_seqs.keys()), round(len(sub_seqs) * val_ratio))
        metadata_val = {k: sub_seqs[k] for k in val_samples}
        metadata_train = {k: sub_seqs[k] for k in sub_seqs if k not in val_samples}

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
        # new_full_ = metadata_val_path.replace('val', 'full')
        # with open(new_full_, 'wb') as fp:
        #     pkl.dump(full_metadata, fp)
        # print(metadata_val_path)

if __name__ == "__main__":
    tyro.cli(build_dl3dv_metadata)