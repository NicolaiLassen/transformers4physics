
from typing import Dict

from omegaconf import DictConfig
from torch import embedding


def sweep_decorate_config(cfg: DictConfig, sweep_params: Dict):
    # cfg.learning.sched = sweep_params["learning.sched"]
    # cfg.learning.lr = sweep_params["learning.lr"]
    # cfg.learning.batch_size_train = int(sweep_params["learning.batch_size_train"])
    
    cfg.embedding.backbone = sweep_params["embedding.backbone"]
    cfg.embedding.backbone_dim = sweep_params["embedding.backbone_dim"]
    cfg.embedding.fc_dim = sweep_params["embedding.fc_dim"]
    cfg.embedding.embedding_dim = sweep_params["embedding.embedding_dim"]