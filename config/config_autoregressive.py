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
        self.embedding_dim = cfg.embedding_dim if cfg.embedding_dim else 128

        self.n_ctx = cfg.n_ctx if cfg.n_ctx else 16
        self.n_layer = cfg.n_layer if cfg.n_layer else 6
        self.n_head = cfg.n_head if cfg.n_head else 4

        self.resid_pdrop = 0.0
        self.embd_pdrop = 0.0
        self.attn_pdrop = 0.0

        self.layer_norm_epsilon = 1e-5
        self.initializer_range = 0.01

        self.output_hidden_states = cfg.output_hidden_states if cfg.output_hidden_states else False
        self.output_attentions = cfg.output_attentions if cfg.output_attentions else False
