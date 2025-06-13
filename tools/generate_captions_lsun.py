import os
# Set the HF_ENDPOINT environment variable
import platform 
if platform.system() == 'Linux' and 'Ubuntu' in platform.version():
     # work on localworkstation
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
else: # work on aistudio k8s cluster -> centos instead of ubuntu
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_HOME'] = '/input/yyb/.cache/huggingface'
    os.environ['HF_HUB_CACHE'] = '/input/yyb/.cache/huggingface/hub'
 

import torch
from accelerate import Accelerator
import hydra
import torch
import os
import random
# import tyro
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from torchvision.transforms import Resize

import hydra
# import tyro
# import omegaconf
from omegaconf import DictConfig, OmegaConf
# from omegaconf.omegaconf import open_dict
from torch.utils.data import Dataset
torch.backends.cudnn.benchmark = True

import importlib
def import_str(string):
    # From https://github.com/CompVis/taming-transformers
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


class DummyDataset(Dataset):
    def __init__(self, base_folder: str):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        self.num_samples = 100
        # Define the path to the folder containing video frames
        self.base_folder = base_folder
        self.img_list = os.listdir(self.base_folder)
        self.channels = 3
        self.resolution = 490

    def fetch_file(self, root_dir, filename):
        """Shortcut to reader's `fetch_file()`."""
        with open(os.path.join(root_dir, filename), 'rb') as f:
            return Image.open(f).convert('RGB')

    def crop_img(self, img):
        width, height = img.size
        shorted_side = min(width, height)
        if shorted_side == width:
            left = 0
            right = shorted_side
            top = (height - shorted_side) // 2
            bottom = (height + shorted_side) // 2
        else:
            left = (width - shorted_side) // 2
            right = (width + shorted_side) // 2
            top = 0
            bottom = shorted_side
        img = img.crop((left, top, right, bottom))
        return img

    def resize_img(self, img, size):
        img = img.resize((size, size), Image.LANCZOS)
        return img
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        # Randomly select a folder (representing a video) from the base folder
        
        # get image
        image_path = self.img_list[idx]
        raw_img = self.fetch_file(self.base_folder, image_path)
        center_img = self.crop_img(raw_img)
        image = self.resize_img(center_img, self.resolution)
        image = np.array(image).astype(np.float32) / 255 * 2.0 - 1.0
        image = image.transpose(2, 0, 1)

        # images = batch['images_mv'][0]
        # if num_input_view == -1:
        #     num_input_view = images.shape[0]
        # images = Resize(490)(images)
        # # parse caption path
        # # run captioning
        # dataset_name, scene_name = batch['dataset_name_mv'][0], batch['scene_name_mv'][0].replace('/','_')
        return {
            'images_mv': image,
            'dataset_name_mv': 'lsun',
            'scene_name_mv': image_path[:-4],
            }


querys = [
    # "<|User|> <ImageHere> This is a view of a scene. Please identify the scene type (object-centric, indoor, outdoor). If it is an outdoor scene, specify whether it is a driving or drone view. Then, provide a detailed description of the scene, ensuring you include details such as weather conditions, lighting conditions",
    "<|User|> <ImageHere> This is a view of a scene, provide a detailed description of the scene, ensuring you include details such as weather conditions, lighting conditions.",
    "<|User|> Please compress the description into three or four sentences. The details about the visual contents, features, object relationships and background information must be retained. Please remove the irrelevant and unnecessary comments, contents that are not related to the object. \
        Remove all words like 'The image' or 'The scene' in your description.",
    # "<|User|> Please further compress the description into two or three sentences. The details about the main visual contents and apperance features must be retained. Please remove the irrelevant and unnecessary comments, contents that are not related to the object.\
    # Remove all words like 'The image' or 'The scene' in your description.",
    "<|User|> Please further compress the description into one or two sentence (no more than 77 words). The main visual contents must be retained. Please remove the irrelevant and unnecessary comments, contents that are not related to the object. You should use different words if possible, campared to your previous responses. \
        Remove all words like 'The image' or 'The scene' in your description.",
    # "<|User|> Please further compress the description into one sentence. The main visual contents must be retained. Please remove the irrelevant and unnecessary comments, contents that are not related to the object. You should use different words if possible, campared to your previous responses.\
    # Remove all words like 'The image' or 'The scene' in your description.",
]



@torch.no_grad()
def generate_single_caption(
        model, 
        tokenizer,
        batch,
        caption_outdir,
        num_input_view = -1,
        force_regen = False):
        
        images = batch['images_mv']
        if num_input_view == -1:
            num_input_view = images.shape[0]
        images = Resize(490)(images)
        # parse caption path
        # run captioning
        dataset_name, scene_name = batch['dataset_name_mv'][0], batch['scene_name_mv'][0].replace('/','_')
        os.makedirs(caption_outdir, exist_ok=True)
        caption_path = os.path.join(caption_outdir, f'{scene_name}.txt')
        if os.path.exists(caption_path) and (not force_regen):
            return True
        results = []
        # for i in range(num_input_view):
        history = ''
        image = (images+ 1) / 2
        for j, query in enumerate(querys):
            history += query
            with torch.amp.autocast("cuda"):
                response, _ = model.chat(tokenizer, 
                                            query=history, 
                                            image=image, 
                                            history=[], 
                                            do_sample=True)
            history += " <|Bot|>" + response
            if j > 0:
                results += [response]

        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results))

        return True


def build_model(use_4bit = False, hf_path = '', device = 'cuda:0'):
    if use_4bit:
        from transformers import AutoModel
        import auto_gptq
        from auto_gptq.modeling import BaseGPTQForCausalLM
        from transformers import AutoModelForCausalLM, AutoTokenizer
        auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
        class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
            layers_block_name = "model.layers"
            outside_layer_modules = [
                'vit', 'vision_proj', 'model.tok_embeddings', 'model.norm', 'output', 
            ]
            inside_layer_modules = [
                ["attention.wqkv.linear"],
                ["attention.wo.linear"],
                ["feed_forward.w1.linear", "feed_forward.w3.linear"],
                ["feed_forward.w2.linear"],
            ]
        # pip install autoawq  torchvision==0.18.1 torch=2.3.1 xformers=0.27.0
        model_path = os.path.join(hf_path, 'internlm/internlm-xcomposer2-vl-7b-4bit')
        model = InternLMXComposer2QForCausalLM.from_quantized(
        model_path, trust_remote_code=True,use_marlin=True, device=device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model_path = os.path.join(hf_path, 'internlm/internlm-xcomposer2d5-7b')
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


@hydra.main(
    version_base=None,
    config_path="configurations",
    config_name="config",
)
def run(cfg: DictConfig):

    # Prepare DDP
    accelerator = Accelerator()
    device = accelerator.device
    # device = "cuda:0"
    # Build internlmx model
    cfg.output_dir = 'data/lsun/'
    model, tokenizer = build_model(
        use_4bit=True, hf_path=cfg.HF_PATH, device = device)
    cfg.image_size = 490
    if cfg.dataset_name == 'lsun':
        training_dataset = DummyDataset(base_folder='data/lsun/images')
    caption_outdir = os.path.join(cfg.output_dir, 'captions')
    data_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False,
        persistent_workers=True)
    model.requires_grad_(False).eval()
    model = model.to(device)
    model.vision_proj = model.vision_proj.to(device)
    #model.model.tok_embeddings = model.model.tok_embeddings.to(device)
    data_loader = accelerator.prepare(data_loader)

    for batch in tqdm(data_loader):
        generate_single_caption(model = model, tokenizer=tokenizer, batch=batch, caption_outdir = caption_outdir, num_input_view = -1)
        

if __name__ == "__main__":
    run()



 

#  accelerate launch --multi_gpu --gpu_ids 0,1,2,3,4,5,6,7 --main_process_port=9910 --num_processes 8 generate_captions_mp.py global_env=aistudio_k8s dataset=mv_full_dataset dataset_name=re10k
# accelerate launch  --main_process_port=9910 --num_processes 1 generate_captions_mp.py global_env=aistudio_k8s dataset=mv_full_dataset dataset_name=lsun