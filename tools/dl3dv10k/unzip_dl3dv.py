"""
Author: yyyyb
Last modified: 2024-08-16
Unzip full DL3DV that download from HF
"""
import os
import zipfile
import tyro
from tqdm import tqdm


def unzip_all_files(
    source_dir: str ='/nas7/datasets/DL3DV-10K-part2/DL3DV-10K',
    target_dir: str = '/nas6/yyb/DL3DV-10K'
    ):
    """XX"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    source_dir_1 = source_dir + '/all_scenes_zip'
    for filename in tqdm(os.listdir(source_dir_1)):
        if filename.endswith('.zip'):
            zip_path = os.path.join(source_dir_1, filename)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print(f"Uizip {zip_path} to {target_dir}")

    source_dir_2 = source_dir + '/all_scenes_zip_part2'
    for filename in tqdm(os.listdir(source_dir_2)):
        if filename.endswith('.zip'):
            zip_path = os.path.join(source_dir_2, filename)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                print(f"Uizip {zip_path} to {target_dir}")
            except Exception as e:
                print(f"SKip {zip_path}")

if __name__ == "__main__":
    tyro.cli(unzip_all_files)