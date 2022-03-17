
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

    def __init__(self, **kwargs) -> None:
        self.image_dim = kwargs.pop("image_dim", (32, 32))
        self.backbone = kwargs.pop("backbone", "conv")
        
        self.embed_dim = kwargs.pop("embed_dim", 128)  
