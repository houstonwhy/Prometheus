import argparse
import json
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import torch

from kiui.cli.clip_sim_text import CLIP


import ipdb
st=ipdb.set_trace

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help="input directory")

    args = parser.parse_args()

    clip = CLIP('cuda', model_name='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')

    input_dir = Path(args.input_dir)
    if 'gaussiandreamer' in str(input_dir):
        input_dir = input_dir / "gaussiandreamer-sd"
    dir_list = list(input_dir.glob('*'))
    
    all_results = []
    for video_dir in tqdm(dir_list):
        if video_dir.is_dir():
            text_prompt = video_dir.name.replace('_', ' ')

            with torch.no_grad():
                ref_features = clip.encode_text(text_prompt)
            
            if 'gaussiandreamer' in str(video_dir):
                images_dir = video_dir / "save" / "it1200-test"
                method = 'gaussiandreamer'
            elif 'lgm' in str(video_dir):
                images_dir = video_dir / video_dir.name
                method = 'lgm'
            elif 'director3d' in str(video_dir):
                images_dir = video_dir / "0" / video_dir.name
                method = 'director3d'
            elif 'prometheus' in str(video_dir):
                images_dir = video_dir / "0" / video_dir.name
                method = f'prometheus_{input_dir.parent.name}'
            else:
                raise ValueError(f"Unknown video directory: {video_dir}")
            images_list = list(images_dir.glob('*'))

            results = []
            for image_path in tqdm(images_list):
                with torch.no_grad():
                    try:
                        cur_features = clip.encode_image(Image.open(image_path))
                    except:
                        continue
                similarity = (ref_features * cur_features).sum(dim=-1).mean().item()
                results.append(similarity)

            all_results.append(np.mean(results))
    
    average_similarity = np.mean(all_results)
    print(f"{method} Average CLIP score: {average_similarity}")
    
    output_metrics = {'average_CLIP_score': average_similarity, "each_CLIP_score": all_results}
    with open(input_dir / 'clip_score.json', 'w') as f:
        json.dump(output_metrics, f, indent=4)