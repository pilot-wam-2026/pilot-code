"""PILOT 340000 framework factory."""
from starVLA.model.tools import FRAMEWORK_REGISTRY


def build_framework(cfg):
    if cfg.framework.name != "CosmoPredict25PerceiverVJEPA2AC":
        raise ValueError("This release supports only the PILOT 340000 architecture")
    from .WAM_VJEPA.CosmoPredict25PerceiverVJEPA2AC import CosmoPredict25_Perceiver_VJEPA2AC
    return CosmoPredict25_Perceiver_VJEPA2AC(config=cfg)
