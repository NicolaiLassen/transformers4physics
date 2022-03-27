from .phys_config import PhysConfig



class AutoregressiveConfig(PhysConfig):
    """Parent class for physical transformer configuration.
    This is a slimmed version of the pretrainedconfig from the Hugging Face
    repository.
    Args:

    Raises:
        AssertionError: If provided parameter is not a config parameter
    """
    model_type: str = " Autoregressive Model Config"

    def __init__(self,
                 cfg,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.activation_function = cfg.activation_function if cfg.activation_function else "gelu_new"
        self.embedding_dim = cfg.n_embedding if cfg.n_embedding else 128

        self.n_ctx = 0
        self.n_layer = 0
        self.n_head = 0
