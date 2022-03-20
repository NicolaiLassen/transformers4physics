
from typing import Dict

from omegaconf import DictConfig
from torch import embedding


def sweep_decorate_config(cfg: DictConfig, sweep_params: Dict):
    cfg.learning.sched = sweep_params["learning.sched"]
    cfg.learning.learning_rate = sweep_params["learning.lr"]
    cfg.learning.epochs = sweep_params["learning.epochs"]
    cfg.learning.batch_size_train = sweep_params["batch_size_train"]
    cfg.opt.name = sweep_params["opt.name"]

    if cfg.train_embedding:
        cfg.embedding.backbone = sweep_params["embedding.backbone"]
        cfg.embedding.backbone_dim = sweep_params["embedding.backbone_dim"]
        cfg.embedding.fc_dim = sweep_params["embedding.fc_dim"]
        cfg.embedding.embedding_dim = sweep_params["embedding.embedding_dim"]
    else:
        cfg.autoregressive.model = sweep_params["autoregressive.model"]

    return cfg
