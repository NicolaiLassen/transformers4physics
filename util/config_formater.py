
from typing import Dict

from omegaconf import DictConfig
from torch import embedding


def sweep_decorate_config(cfg: DictConfig, sweep_params: Dict):
    cfg.learning.sched = sweep_params["learning__sched"]
    cfg.learning.learning_rate = sweep_params["learning__learning_rate"]
    cfg.opt.name = sweep_params["opt__name"]

    if cfg.train_embed:
        cfg.embedding.backbone = sweep_params["embedding__backbone"]
        cfg.embedding.ae_dim = sweep_params["embedding__ae_dim"]
        cfg.embedding.embed_dim = sweep_params["embedding__embed_dim"]
    else:
        cfg.autoregressive.model = sweep_params["autoregressive__model"]

    return cfg
