
from typing import Dict

from omegaconf import DictConfig
from torch import embedding


def sweep_decorate_config(cfg: DictConfig, sweep_params: Dict):
    cfg.learning.sched = sweep_params["learning.sched"]
    cfg.learning.learning_rate = sweep_params["learning.lr"]
    cfg.opt.name = sweep_params["opt.name"]
    cfg.batch_size = sweep_params["batch_size"]

    if cfg.train_embed:
        cfg.embedding.backbone = sweep_params["embedding.backbone"]
        cfg.embedding.backbone_dim = sweep_params["embedding.backbone_dim"]
        cfg.embedding.fc_dim = sweep_params["embedding.fc_dim"]
        cfg.embedding.embedding_dim = sweep_params["embedding.embedding_dim"]
    else:
        cfg.autoregressive.model = sweep_params["autoregressive.model"]

    return cfg
