"""
Author: yyyyb
Last modified: 2024-09-03
Build metadata of Co3Dv2 for prometheus/director3d training
"""

import os
import os.path as osp
import tyro
from tqdm import tqdm
import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader

import gzip
import json
from collections import defaultdict
from functools import cache
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Literal
from pathlib import Path
TRAINING_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]

TEST_CATEGORIES = [
    "ball",
    "book",
    "couch",
    "frisbee",
    "hotdog",
    "kite",
    "remote",
    "sandwich",
    "skateboard",
    "suitcase",
]
FULL_CATEGORIES = TRAINING_CATEGORIES + TEST_CATEGORIES
assert len(TRAINING_CATEGORIES) + len(TEST_CATEGORIES) == 51

Frame = tuple[int, Path]
@dataclass
class Sequence:
    name: str
    category: str
    frames: list[Frame]
    viewpoint_quality_score: float | None

# Set cwd to current file dir instead of workspace folder
current_file_path = osp.abspath(__file__)
current_directory = osp.dirname(current_file_path)
os.chdir(current_directory)

def fetch_txt(path):
    with open(path, 'r') as f:
        file_bytes = f.read()
    return file_bytes
class Co3DDataset(Dataset):
    # CO3D dataset partly borrow from
    def __init__(self, root_dir, categories='', use_caption=True):
        self.root_dir = Path(root_dir)
        self.use_caption = use_caption
       # self.scene_list = self.load_or_parse_scene_list(root_dir, category)
        self.categories = categories
        self.load_cameras = True
        self.sequences = []
        self.load_sequences()
        print('?')
        
    def load_or_parse_scene_list(self, dataset_root, category):
        scene_list_txt = f'mvimgnet_scenes_{category}.txt'
        if osp.isfile(scene_list_txt):
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
        anno_list = []
        for category in categories:
            if not osp.isdir(osp.join(dataset_root, category)):
                continue
            for idx in os.listdir(osp.join(dataset_root, category)):
                if not osp.isdir(osp.join(dataset_root, category, idx)):
                    continue
                scene_list.append(osp.join(dataset_root, category, idx))
        return scene_list

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item =self.sequences[idx]
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
        
    def load_sequences(self):
        num_skipped = 0
        categories = [
            dir
            for dir in self.root_dir.iterdir()
            if dir.is_dir() and (dir.name in FULL_CATEGORIES)
        ]

        # Only load configuration-defined categories.
        if self.categories is not None:
            categories = [
                category
                for category in categories
                if category.name in self.categories
            ]

        for category_root in tqdm(categories, desc="Loading CO3D sequences"):
            # Read the set list.
            sequences = self.load_category_sequences(category_root)

            # Filter out sequences with incomplete camera information.
            if self.load_cameras:
                annotations = self.load_frame_annotations(category_root.name)
                for sequence in sequences:
                    for index, _ in sequence.frames:
                        if annotations.get(sequence.name, []).get(index, None) is None:
                            num_skipped += 1
                            break
                    else:
                        self.sequences.append(sequence)
                print(
                    f"[CO3D] Skipped {num_skipped} sequences. Kept "
                    f"{len(self.sequences)} sequences."
                )
            else:
                self.sequences.extend(sequences)

    def load_category_sequences(self, category_path: Path):
        # Instead of reading the set lists, just load all frames.
        sequences = {}
        for example in category_path.iterdir():
            # Skip files and folders that aren't examples.
            if not example.is_dir() or not (example / "images").exists():
                continue

            sequence = []
            for frame in sorted((example / "images").iterdir()):
                assert frame.name.startswith("frame") and frame.suffix == ".jpg"
                index = int(frame.stem[5:])
                sequence.append((index, frame))

            sequences[example.name] = sequence

        # Generate sequence structs.
        sequences = [
            Sequence(name, category_path.name, frames, None)
            for name, frames in sequences.items()
        ]

        # Load the sequence annotations.
        sequence_annotations = json.loads(
            gzip.GzipFile(category_path / "sequence_annotations.jgz", "rb")
            .read()
            .decode("utf8")
        )
        sequence_annotations = {
            annotation["sequence_name"]: annotation
            for annotation in sequence_annotations
        }

        # Add viewpoint quality scores.
        valid_sequences = []
        for sequence in sequences:
            annotations = sequence_annotations[sequence.name]
            score = annotations.get("viewpoint_quality_score", None)
            if score is not None:
                sequence.viewpoint_quality_score = score
                sequence.frames = sorted(sequence.frames)
                valid_sequences.append(sequence)

        return valid_sequences
    
    @cache
    def load_frame_annotations(self, category: str):
        frame_annotations = json.loads(
            gzip.GzipFile(self.root_dir / category / "frame_annotations.jgz", "rb")
            .read()
            .decode("utf8")
        )

        annotations = defaultdict(dict)

        # Extract camera parameters.
        for frame_annotation in frame_annotations:
            sequence = frame_annotation["sequence_name"]
            frame = frame_annotation["frame_number"]
            annotations[sequence][frame] = {
                **frame_annotation["viewpoint"],
                **frame_annotation["image"],
            }

        return dict(annotations)

    def read_camera_parameters(
        self,
        sequence: Sequence,
        frame_index_in_sequence: int,
    ):
        annotations = self.load_frame_annotations(sequence.category)

        index, _ = sequence.frames[frame_index_in_sequence]
        annotation = annotations[sequence.name][index]

        # Process the intrinsics.
        p = annotation["principal_point"]
        f = annotation["focal_length"]
        h, w = annotation["size"]
        assert annotation["intrinsics_format"] == "ndc_isotropic"
        k = torch.eye(3, dtype=torch.float32)
        s = min(h, w) / 2
        k[0, 0] = f[0] * s
        k[1, 1] = f[1] * s
        k[0, 2] = -p[0] * s + w / 2
        k[1, 2] = -p[1] * s + h / 2
        k[:2] /= torch.tensor([w, h], dtype=torch.float32)[:, None]

        # Process the extrinsics.
        w2c = torch.eye(4, dtype=torch.float32)
        w2c[:3, :3] = torch.tensor(annotation["R"], dtype=torch.float32).T
        w2c[:3, 3] = torch.tensor(annotation["T"], dtype=torch.float32)
        flip_xy = torch.diag_embed(torch.tensor([-1, -1, 1, 1], dtype=torch.float32))
        w2c = flip_xy @ w2c
        c2w = w2c.inverse()

        return c2w, k

def build_co3d_metadata(
        #root_dir: str = '/nas6/yyb/MVImgNet',
        root_dir: str = '/nas1/datasets/CO3D_V2',
        output_dir: str = '/input/yyb/PlatonicGen/metadatas/MVImgNet',
        use_caption: bool = False,
        category: str = 'apple',
        mode: str = 'full',
        num_workers: int = 16,
        batch_size: int = 32
):
    """Build metadata for mvimgnet dataset"""

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
    dataset = Co3DDataset(root_dir, [category], use_caption)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x)

    metadata = {}
    for batch in tqdm(dataloader):
        for scene_meta in batch:
            if scene_meta is not None:
                metadata[scene_meta['name']] = scene_meta
    # further filter by if has captions
    # if use_caption:
    #     metadata = {k: v for k, v in metadata.items() if 'captions' in v.keys()}

    metadata_path = osp.join(output_dir, 'metadata', f'co3d_{mode}.pkl')
    os.makedirs(osp.join(output_dir, 'metadata'), exist_ok=True)
    with open(metadata_path, 'wb') as fp:
        pkl.dump(metadata, fp)

if __name__ == "__main__":
    # tyro.cli(build_mvimgnet_metadata)
    tyro.cli(build_co3d_metadata)