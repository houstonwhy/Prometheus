import argparse
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import json
from brisque import BRISQUE

import ipdb
st=ipdb.set_trace

if __name__ == "__main__":
    input_dir = "/home/jiahao/workspace/LGM/outputs/director3d/prompt_single/A_sparkling_diamond_tiara"
    obj = BRISQUE(url=False)

    input_dir = Path(input_dir)
    dir_list = [input_dir]

    all_results = []
    for video_dir in tqdm(dir_list):
        if video_dir.is_dir():
            if 'gaussiandreamer' in str(video_dir):
                images_dir = video_dir / "save" / "it1200-test"
                method = 'gaussiandreamer'
            elif 'lgm' in str(video_dir):
                images_dir = video_dir / video_dir.name
                method = 'lgm'
            elif 'director3d' in str(video_dir):
                images_dir = video_dir / "0" / video_dir.name
                method = 'director3d'
            else:
                raise ValueError(f"Unknown video directory: {video_dir}")
            images_list = list(images_dir.glob('*'))

            results = []
            for image_path in tqdm(images_list, desc=f"Processing {video_dir.name}"):
                try:
                    image = np.array(Image.open(image_path))
                except:
                    continue
                metric = obj.score(image)
                if np.isnan(metric):
                    print(f"NaN found in {image_path}")
                    continue
                results.append(metric)
            all_results.append(np.mean(results))
    
    average_niqe = np.mean(all_results)
    print(f"{method} Average BRISQUE: {average_niqe}")
    
    output_metrics = {'average_BRISQUE': average_niqe, 'all_results': all_results}
    with open(input_dir / 'BRISQUE.json', 'w') as f:
        json.dump(output_metrics, f, indent=4)