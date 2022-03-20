
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
                 channels=3,
                 image_dim=32,
                 backbone="TwinsSVT",
                 backbone_dim=64,
                 embedding_dim=128,
                 fc_dim=128,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.channels = channels,
        self.image_dim = image_dim
        self.backbone = backbone
        self.backbone_dim = backbone_dim
        self.embedding_dim = embedding_dim
        self.fc_dim = fc_dim
