"""
Author: yyyyb
Last modified: 2024-08-17
Build metadata of SAM1B+LLaVA captions(~11M) for prometheus/director3d training
Captiondata get from https://huggingface.co/datasets/PixArt-alpha/SAM-LLaVA-Captions10M
"""
import os
import os.path as osp
import glob
import tyro
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv

class TextDataset(Dataset):
    """_summary_

    _extended_summary_

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, root_dir, use_caption = False, tiny = False,):
        self.root_dir = root_dir
        self.tiny = tiny
        self.use_caption = use_caption
        self.filename_list = []
        if not tiny:
            with open(osp.join(root_dir, 'SAM_1B_filenames.txt'), 'r', encoding='utf-8') as ff:
                self.filename_list = ff.read().split('\n')
        else:
            image_list_ =  glob.glob(osp.join(root_dir, '**', f'*.jpg'), recursive=True)
            self.filename_list = [osp.relpath(ff, root_dir) for ff in image_list_]
        print('')

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        filename = self.filename_list[idx]
        image_name = filename.split('/')[-1]
        caption_file_path = osp.join(self.root_dir, 'captions', image_name.replace('.jpg', '.txt'))
        if self.use_caption:
            try:
                #print(filename)
                with open(caption_file_path, 'r') as caption_file:
                    caption_content = caption_file.read().strip()
                return image_name, filename, caption_content
            except Exception as e:
                print(f"Warning: Caption file {image_name} can not open, skip it")
                return None, None, None
        else:
            return image_name, filename, ''


def collate_fn(batch):
    """XX"""
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def generate_imgcaption_csv(root_dir: str = '/data1/yyb/datasets/SAM_1B',
                            output_dir: str = '/data1/yyb/PlatonicGen/data/SAM_1B/local_tiny',
                             num_workers: int = 16,
                             batch_size: int = 32,
                             use_caption: bool = False,
                             mode: str = 'full'
                             ):
    """XX"""
    dataset = TextDataset(root_dir, use_caption = use_caption, tiny = ('tiny' in output_dir))
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            prefetch_factor=4
                            )

    row_count = 0
    file_count = 0
    csvfile = None
    csvwriter = None
    if not use_caption:
        mode += '_nocaption'
    os.makedirs(osp.join(output_dir, 'metadata'), exist_ok=True)

    for batch in tqdm(dataloader):
        if row_count > 500000 or row_count == 0 :
            if csvfile:
                csvfile.close()
            csv_file_path = osp.join(output_dir, 'metadata', f'filenames_and_captions_{mode}_{file_count}.csv')
            csvfile = open(csv_file_path, 'w', encoding='utf-8', newline='')
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Image Name', 'Image Path', 'Caption'])  # headline
            file_count += 1
            row_count = 0

        for image_name, filename, caption_content in zip(batch[0], batch[1], batch[2]):
            csvwriter.writerow([image_name, filename, caption_content])
            row_count += 1

    if csvfile:
        csvfile.close()

    print(f"Image-Caption CSV files have been generated in {root_dir}")

if __name__ == "__main__":
    tyro.cli(generate_imgcaption_csv)