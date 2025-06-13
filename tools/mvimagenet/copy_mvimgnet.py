import shutil
import os
from tqdm import tqdm

# 源目录和目标目录
source_base = '/nas1/datasets/MVImgNet/mvi_all'
destination = '/nas6/yyb/MVImgNet'

# 创建目标目录（如果不存在）
if not os.path.exists(destination):
    os.makedirs(destination)

# 获取所有 mvi_{:02d} 目录
mvi_dirs = [d for d in os.listdir(source_base) if os.path.isdir(os.path.join(source_base, d)) and d.startswith('mvi_')]

# 遍历每个 mvi_{:02d} 目录
for mvi_dir in mvi_dirs:
    source_dir = os.path.join(source_base, mvi_dir)
    # 获取 mvi_{:02d} 目录下的所有子目录
    sub_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d)) and d.isdigit()]
    for sub_dir in tqdm(sub_dirs, desc=f"Copying {mvi_dir} directories", unit="directory") :
        ids = os.listdir(os.path.join(source_dir, sub_dir))
        for scene_id in ids:
            source_sub_dir = os.path.join(source_dir, sub_dir, scene_id)
            destination_sub_dir = os.path.join(destination, sub_dir, scene_id)
            # 检查目标目录中是否已经存在同名的场景目录
            if not os.path.exists(destination_sub_dir):
                # 使用 shutil.copytree 复制目录
                shutil.copytree(source_sub_dir, destination_sub_dir)

    print(f"Copying {mvi_dir} completed!")