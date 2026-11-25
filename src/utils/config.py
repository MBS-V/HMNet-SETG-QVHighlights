from pathlib import Path
from omegaconf import OmegaConf

def load_config(path='configs/default.yaml',overrides=None):
    path=Path(path)
    if path.name!='default.yaml' and (path.parent/'default.yaml').exists():
        cfg=OmegaConf.merge(OmegaConf.load(path.parent/'default.yaml'),OmegaConf.load(path))
    else:
        cfg=OmegaConf.load(path)
    if overrides:
        for key,value in overrides.items(): OmegaConf.update(cfg,key,value,merge=True)
    OmegaConf.resolve(cfg)
    return cfg
