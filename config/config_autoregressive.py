from config.phys_config import PhysConfig



class AutoregressiveConfig(PhysConfig):
    """Parent class for physical transformer configuration.
    This is a slimmed version of the pretrainedconfig from the Hugging Face
    repository.
    Args:

    Raises:
        AssertionError: If provided parameter is not a config parameter
    """
    model_type: str = ""

    def __init__(self, **kwargs) -> None:
        print(kwargs)
