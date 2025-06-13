"""
Author: yyyyb
Last modified: 2024-08-16
Generte camera poses of MVImgNet from clomap project files
Code borrow from https://github.com/Fyusion/LLFF/tree/master/llff/poses
"""
import os
import os.path as osp
import json
import random
import tyro
from tqdm.auto import tqdm
# Set cwd to current file dir instead of workspace folder
current_file_path = osp.abspath(__file__)
current_directory = osp.dirname(current_file_path)
os.chdir(current_directory)

from poses.pose_utils import gen_poses

def parse_scene_list(dataset_root:str,
                    category:str = 'full',
                    save_txt = True
                    ):
    scene_list = []
    if category == 'all':
        categories = os.listdir(dataset_root)
    else:
        categories = [category]

    for category in tqdm(categories):
        if not osp.isdir(osp.join(dataset_root, category)):
            continue
        for idx in os.listdir(osp.join(dataset_root, category)):
            if not osp.isdir(osp.join(dataset_root, category, idx)):
                continue
            scene_list.append(osp.join(category, idx))
    # if save_txt:
    #     pass
    return scene_list
def generate_cameras(
    dataset_root : str = "/data1/yyb/datasets/MVImgNet",
    category : str = '99',
    force: bool =  True
):
    """XD
    Args:
        dataset_root (str, optional): _description_. Defaults to "/nas6/yyb/MVImgNet".
        category (str, optional): _description_. Defaults to 'all'.
    """
    basedirs = []
    scene_list_txt = f'mvimgnet_scenes_{category}_.txt'
    if osp.isfile(scene_list_txt):
        with open(scene_list_txt, 'r') as tx:
            basedirs = tx.read().split('\n')
    else:
        basedirs = parse_scene_list(dataset_root, category)
    basedirs = [item.strip("/") for item in basedirs]
    #random.shuffle(basedirs)
    for scene in tqdm(basedirs):
        poses_path = osp.join(dataset_root, scene, 'cameras.json')
        if osp.exists(poses_path) and (not force):
            print(f'{scene} exist cameras.json, skip')
            continue
        dict2save = {}
        try:
            poses, perm = gen_poses(basedir=osp.join(dataset_root, scene))
        except:
            print(f'{scene} has no colmap pose')
            continue
        if poses is None:
            print(f'{scene} has no colmap pose')
            continue
        for i, idx in enumerate(perm):
            dict2save[int(idx)] = {
            'c2w':poses[:, :4, i].tolist(),
            'hwf':poses[:, 4, i].tolist()
            }
        #json2save = json.loads(dict2save)
        with open(poses_path, 'w') as json_file:
            json.dump(dict2save, json_file)
        # print('??')

if __name__ == "__main__":
    tyro.cli(generate_cameras)