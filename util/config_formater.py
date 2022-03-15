
import re
from typing import Dict

from omegaconf import DictConfig


def unpack_model_tree(model_tree: str):
    return {}

def get_model_from_backbone(backbone: str, model_trees: str):
    for model_tree in model_trees.split(","):
        if backbone in model_tree:
            return model_tree

def sweep_decorate_config(cfg: DictConfig , sweep_params: Dict):
    cfg.learning.sched = sweep_params["learning__sched"]
    cfg.learning.learning_rate = sweep_params["learning__learning_rate"]
    cfg.opt.name = sweep_params["opt__name"]  

    if cfg.train_embed:
        backbone = sweep_params["embedding__backbone"]
        cfg.embedding.backbone = backbone
        cfg.embedding.observable_net = get_model_from_backbone(backbone, sweep_params["embedding__observable_net"])
        
    else:
        cfg.autoregressive.model = sweep_params["autoregressive__model"]

    return cfg
