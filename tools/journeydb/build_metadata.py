"""
Author: yyyyb
Last modified: 2024-10-21
Build metadata of JourneyDB dataset
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

def generate_imgcaption_csv(
        root_dir: str = '/data1/yyb/PlatonicGen/data/JourneyDB/JourneyDB_hw_raw_caption.csv',
        output_dir: str = '/data1/yyb/PlatonicGen/data/JourneyDB/',
        local_data_root: str = '/data1/yyb/datasets/JourneyDB',
        local_tiny: bool = True,
        num_workers: int = 16,
        batch_size: int = 32,
        use_caption: bool = False,
        mode: str = 'full'):
    """XX"""
    
    row_count = 0
    file_count = 0
    csvfile = None
    csvwriter = None
    # for csv_file in csv_paths:
    csv_file = root_dir
    if local_tiny:
        output_dir = os.path.join(output_dir, 'local_tiny')
    
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f)
        for i, line in tqdm(enumerate(csv_reader)):
            # set writer
            if i == 0:
                continue
            if row_count > 100000 or row_count == 0 :
                if csvfile:
                    csvfile.close()
                csv_file_path = osp.join(output_dir, 'metadata', f'filenames_and_captions_{mode}_{file_count}.csv')
                os.makedirs(osp.join(output_dir, 'metadata'), exist_ok=True)
                csvfile = open(csv_file_path, 'w', encoding='utf-8', newline='')
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Image Name', 'Image Path', 'Caption'])  # headline
                file_count += 1
                row_count = 0

            image_name = line[0].split('/')[-1]
            filename = '/'.join(line[0].split('/')[2:])
            if local_tiny and not os.path.exists(os.path.join(local_data_root, filename)):
                continue
            caption_content = line[-1]
            csvwriter.writerow([image_name, filename, caption_content])
            row_count += 1

    if csvfile:
        csvfile.close()

    print(f"Image-Caption CSV files have been generated in {root_dir}")

    # if not use_caption:
    #     mode += '_nocaption'
    # os.makedirs(osp.join(output_dir, 'metadata'), exist_ok=True)

    # for batch in tqdm(dataloader):
    #     if row_count > 500000 or row_count == 0 :
    #         if csvfile:
    #             csvfile.close()
    #         csv_file_path = osp.join(output_dir, 'metadata', f'filenames_and_captions_{mode}_{file_count}.csv')
    #         csvfile = open(csv_file_path, 'w', encoding='utf-8', newline='')
    #         csvwriter = csv.writer(csvfile)
    #         csvwriter.writerow(['Image Name', 'Image Path', 'Caption'])  # headline
    #         file_count += 1
    #         row_count = 0

    #     for image_name, filename, caption_content in zip(batch[0], batch[1], batch[2]):
    #         csvwriter.writerow([image_name, filename, caption_content])
    #         row_count += 1

    # if csvfile:
    #     csvfile.close()

    # print(f"Image-Caption CSV files have been generated in {root_dir}")

if __name__ == "__main__":
    tyro.cli(generate_imgcaption_csv)