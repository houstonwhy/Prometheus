"""Collects all datasets."""
try:
    from pcache_fileio import fileio # pylint: disable=E0401
    # import fsspec
    # PCACHE_HOST = "vilabpcacheproxyi-pool.cz50c.alipay.com"
    # PCACHE_PORT = 39999
    # pcache_kwargs = {"host": PCACHE_HOST, "port": PCACHE_PORT}
    # pcache_fs = fsspec.filesystem("pcache", pcache_kwargs=pcache_kwargs)

except Exception:
    print("pcacheio not install")


from .base_dataset import ProbDataset
from .base_dataset import JointDataset
from .t2i_dataset import Text2ImageDataset
from .mvimgnet_dataset import MVImgNetDataset
from .dl3dv10k_dataset import DL3DV10KDataset
from .urban_dataset import UrbanGenDataset
from .re10k_dataset import RealEstate10KDataset
from .re10k_dataset import RealEstate10KDatasetEval
from .objaverse import ObjaverseDataset

_DATASETS = {
    'JointDataset' : JointDataset,
    'ProbDataset' : ProbDataset,
    'MVImgNetDataset': MVImgNetDataset,
    'DL3DV10KDataset': DL3DV10KDataset,
    'Text2ImageDataset': Text2ImageDataset,
    'UrbanGenDataset': UrbanGenDataset,
    'RealEstate10KDataset' : RealEstate10KDataset,
    'RealEstate10KDatasetEval' : RealEstate10KDatasetEval,
    'ObjaverseDataset' : ObjaverseDataset
}
