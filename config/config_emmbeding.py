
from config.phys_config import PhysConfig


class EmmbedingConfig(PhysConfig):
    """Parent class for physical transformer configuration.
    This is a slimmed version of the pretrainedconfig from the Hugging Face
    repository.
    Args:

    Raises:
        AssertionError: If provided parameter is not a config parameter
    """
    model_type: str = ""

    def __init__(self,
                 cfg,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.channels = cfg.channels if cfg.channels else 3
        self.image_dim = cfg.image_dim if cfg.image_dim else 32
        self.backbone = cfg.backbone if cfg.backbone else "TwinsSVT"
        self.backbone_dim = cfg.backbone_dim if cfg.backbone_dim else 64
        self.embedding_dim = cfg.embedding_dim if cfg.embedding_dim else 128
        self.fc_dim = cfg.fc_dim if cfg.fc_dim else 128