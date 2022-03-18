
from typing import Dict

from omegaconf import DictConfig
from torch import embedding


def sweep_decorate_config(cfg: DictConfig, sweep_params: Dict):
    cfg.learning.sched = sweep_params["learning.sched"]
    cfg.learning.learning_rate = sweep_params["learning.learning_rate"]
    cfg.opt.name = sweep_params["opt.name"]

    if cfg.train_embed:
        cfg.embedding.backbone = sweep_params["embedding.backbone"]
        cfg.embedding.ae_dim = sweep_params["embedding.ae_dim"]
        cfg.embedding.embed_dim = sweep_params["embedding.embed_dim"]
    else:
        cfg.autoregressive.model = sweep_params["autoregressive.model"]

    return cfg
