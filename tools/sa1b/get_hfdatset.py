"""
Author: yyyyb
Last modified: 2024-08-11
Download SAM-LLaVA-Captions10M caption file from huggingface
"""
import os
# Set the HF_ENDPOINT environment variable
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # before import all hf related pkgs
import webdataset as wds
from huggingface_hub import HfFileSystem, get_token, hf_hub_url

fs = HfFileSystem()
files = [fs.resolve_path(path) for path in fs.glob("hf://datasets/PixArt-alpha/SAM-LLaVA-Captions10M/SA1B_caption.tar.gz")]
urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]
urls = f"pipe: curl -s -L -H 'Authorization:Bearer {get_token()}' {'::'.join(urls)}"

ds = wds.WebDataset(urls).decode()


from datasets import load_dataset

ds = load_dataset("PixArt-alpha/SAM-LLaVA-Captions10M")