import copy
import json
import logging
import os
from re import S
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class PhysConfig(object):
    """Parent class for physical transformer configuration.
    This is a slimmed version of the pretrainedconfig from the Hugging Face
    repository.
    Args:
    
    Raises:
        AssertionError: If provided parameter is not a config parameter
    """
    model_type: str = ""

    def __init__(self, **kwargs) -> None:
        self.config_name = kwargs.pop("config_name", "")
        self.pretrained = kwargs.pop("pretrained", False)
        self.ckpt_path = kwargs.pop("ckpt_path", "")

        self.check_configs(kwargs)

    def check_configs(self, **kwargs):
          for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save a configuration object to JSON file.
        Args:
            save_directory (str): Directory where the configuration JSON file will be saved.
        Raises:
            AssertionError: If provided directory does not exist.
        """
        if os.path.isfile(save_directory):
            raise AssertionError("Provided path ({}) should be a directory, not a file".format(save_directory))
        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, self.config_name)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info("Configuration saved in {}".format(output_config_file))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, any], **kwargs) -> "EmmbedingConfig":
        """
        Constructs a config from a Python dictionary of parameters.
        Args:
            config_dict (Dict[str, any]): Dictionary of parameters.
            kwargs (Dict[str, any]): Additional parameters from which to initialize the configuration object.
        Returns:
            (PhysConfig): An instance of a configuration object
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config %s", str(config))
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    def to_dict(self) -> Dict[str, any]:
        """
        Serializes this instance to a Python dictionary.
        Returns:
            (Dict[str, any]): Dictionary of config attributes
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.
        Returns:
            (str): String of configuration instance in JSON format.
        """
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: str) -> None:
        """
        Save config instance to JSON file.
        Args:
            json_file_path (str): Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def update(self, config_dict: Dict) -> None:
        """
        Updates attributes of this class with attributes from provided dictionary.
        Args:
            config_dict (Dict): Dictionary of attributes that shall be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)
