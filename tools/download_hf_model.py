"""
Author: yyyyb
Last modified: 2024-08-11
Download given model/dataset file/repo from huggingface
"""
import os.path as osp
import tyro
from huggingface_hub import snapshot_download, hf_hub_download


def download_from_hf(
        repo_name: str= '',
        filename: str = '',
        repo_type: str = '',
        local_dir: str = '/data0/hf_weights/',
        endpoint: str ='https://hf-mirror.com',
):
    """XX"""
    # Download the full model repository
    if not filename:
        snapshot_download(
            repo_id=repo_name,
            repo_type =None if (not repo_type)  else repo_type,
            force_download=True,
            resume_download=True,
            local_dir=osp.join(local_dir, repo_name),
            local_dir_use_symlinks=False,
            endpoint=endpoint,
            etag_timeout=500
        )
    # Download single file from certrain repo
    else:
        hf_hub_download(
            filename = filename,
            repo_id = repo_name,
            repo_type = None if (not repo_type) else repo_type,
            force_download=True,
            resume_download=True,
            local_dir=osp.join(local_dir, repo_name),
            local_dir_use_symlinks=False,
            endpoint=endpoint,
            etag_timeout=500
        )

if __name__ == "__main__":
    tyro.cli(download_from_hf)
