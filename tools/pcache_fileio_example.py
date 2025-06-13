import os
import os.path as osp
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from pcache_fileio import fileio

'/mnt/antsys-vilab_datasets_pcache_datasets/yyb/DL3DV-10K/b54322e64b2422c80f88160a59b39e7fa18adf703c5c1a23c27a3455ad494c57/transforms.json'

oss_folder_path = 'oss://antsys-vilab/datasets/pcache_datasets/'
oss_sam_folder = 'oss://antsys-vilab/datasets/pcache_datasets/SAM_1B/'

pcache_folder_path = 'pcache://vilabpcacheproxyi-pool.cz50c.alipay.com:39999/mnt/antsys-vilab_datasets_pcache_datasets/'

seq_list= [osp.basename(osp.normpath(item)) for item in os.listdir(oss_sam_folder.replace(oss_folder_path, pcache_folder_path))]

filenames_full = []
for seq in tqdm(seq_list):
    seq_dir = osp.join(pcache_folder_path, 'SAM_1B', seq, 'images')
    filenames = [osp.join(seq, 'images', osp.basename(ff)) for ff in os.listdir(seq_dir) if ff.endswith('.jpg')]
    filenames_full += filenames

output_file = 'SAM_1B_filenames.txt'
with open(output_file, 'w') as f:
    for filename in tqdm(filenames_full):
        # 将元素写入文件，每个元素占一行
        f.write(str(filename) + '\n')

#with open(pcache_file_path, 'rb') as f:
#    image = Image.open(f).convert('RGB')
#    image.save('aaa.png')