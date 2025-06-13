"""Collects all training systems/trainers."""
# from director3d
# from .gm_ldm_system import GMLDMSystem
from .traj_dit_system import TrajDiTSystem
# new
from .gs_decoder_system import GSDecoderSystem
from .mv_ldm_system import MVLDMSystem

_SYSTEMS = {
    "TrajDiTSystem" : TrajDiTSystem,
    # "GMLDMSystem" : GMLDMSystem,
    "GSDecoderSystem": GSDecoderSystem,
    "MVLDMSystem":MVLDMSystem
}
