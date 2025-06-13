import os
import shutil
from PIL import Image

# 源目录和目标目录
source_dir = '/data0/jhshao/workspace/lsun/data'
target_dir = '/data1/yyb/datasets/lsun/'  # 替换为目标文件夹路径

# 确保目标目录存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

def convert_webp_to_jpg_recursively(source_dir, target_dir):
    for entry in os.scandir(source_dir):
        if entry.is_dir():
            # 如果是子目录，递归调用
            convert_webp_to_jpg_recursively(entry.path, target_dir)
        elif entry.is_file() and entry.name.lower().endswith('.webp'):
            # 如果是 .webp 文件，进行格式转换
            source_file = entry.path
            # 设置目标文件路径，替换扩展名为 .jpg
            target_file = os.path.join(target_dir, os.path.splitext(entry.name)[0] + '.jpg')

            try:
                # 打开 .webp 图片
                with Image.open(source_file) as img:
                    # 转换为 RGB 模式，因为 .webp 文件可能是 RGBA，需要转换为 RGB 才能保存为 .jpg
                    img = img.convert('RGB')
                    # 保存为 .jpg 格式
                    img.save(target_file, 'JPEG')
                    print(f"Converted {source_file} to {target_file}")
            except Exception as e:
                print(f"Failed to convert {source_file}: {e}")

# 调用递归函数
convert_webp_to_jpg_recursively(source_dir, target_dir)
