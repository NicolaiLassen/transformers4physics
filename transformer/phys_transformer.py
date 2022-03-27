"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import logging
import os
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import torch
from config.config_autoregressive import AutoregressiveConfig
from config.config_emmbeding import EmmbedingConfig
from .phys_transformer_functions import Conv1D
from torch import nn

logger = logging.getLogger(__name__)

Tensor = torch.Tensor


class Physformer(nn.Module):
    """Parent class for physical transformers
    """
    model_name: str = "transformer_model"

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        # Save config in model
        self.config = config

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def generate(self):
        pass

    def get_input_embeddings(self):
        # TODO: Is this really needed?
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        # TODO: Is this really needed?
        self.wte = new_embeddings

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _num_parameters(self) -> int:
        """Gets number of learnable parameters
        Returns:
            int: Number of parameters
        """
        count = 0
        for name, param in self.named_parameters():
            # print(name, param.numel())
            count += param.numel()
        return count

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None:
            self._tie_or_clone_weights(
                output_embeddings, self.get_input_embeddings())

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """ Tie or clone module weights depending of whether we are using TorchScript or not
        """
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(
                input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = torch.nn.functional.pad(
                output_embeddings.bias.data,
                (0, output_embeddings.weight.shape[0] -
                 output_embeddings.bias.shape[0],),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def save_model(self, save_directory: str, epoch: int = 0) -> None:
        """Saves transformer model to the specified directory.
        Args:
            save_directory (str): Folder to save file at
            epoch (int, optional): Epoch number to name model file. Defaults to 0.
        Raises:
            AssertionError: If provided directory is not valid.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(
                "Provided path ({}) should be a directory, not a file".format(save_directory))

        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(
            save_directory, "{}{:d}.pth".format(self.model_name, epoch))
        # Save pytorch model to file
        torch.save(self.state_dict(), output_model_file)

    def load_model(self, file_or_path_directory: str, epoch: int = 0) -> None:
        """Load a transformer model from the specified file or path

        Args:
            file_or_path_directory (str): File or folder path to load state dictionary from.
            epoch (int, optional): Epoch of current model for file name, used if folder path is provided. Defaults to 0.

        Raises:
            FileNotFoundError: If provided file or directory could not be found.
        """
        if os.path.isfile(file_or_path_directory):
            logger.info('Loading embedding model from file: {}'.format(
                file_or_path_directory))
            self.load_state_dict(torch.load(
                file_or_path_directory, map_location=lambda storage, loc: storage))
        elif os.path.isdir(file_or_path_directory):
            file_path = os.path.join(
                file_or_path_directory, "{}{:d}.pth".format(self.model_name, epoch))
            logger.info(
                'Loading embedding model from file: {}'.format(file_path))
            self.load_state_dict(torch.load(
                file_path, map_location=lambda storage, loc: storage))
        else:
            raise FileNotFoundError(
                "Provided path or file ({}) does not exist".format(file_or_path_directory))


class PhysformerTrain(Physformer):
    """Model head for training the physics transformer base.
    Args:
        config (PhysConfig): Phys-transformer config object
        transformer_model (PhysformerBase): Initialized transformer model
    """

    def __init__(self, config: AutoregressiveConfig, transformer_model: Physformer = None) -> None:
        """Constructor
        """
        super().__init__(config)
        self.transformer = transformer_model
        self.transformer.apply(self._init_weights)

    def forward(
        self,
        inputs_embeds: Tensor,
        labels_embeds: Tensor,
        **kwargs
    ) -> Tuple[Union[float, Tensor]]:
        """Forward method for this head calculates the MSE between the predicted time-series and target embeddings
        This head allows for easy distribution to multiple GPUs and CPUs. See transformer 
        Args:
            inputs_embeds (Tensor): [B, T, n_embed] Input features
            labels_embeds (Tensor): [B, T, n_embed] Target output features
            **kwargs (optional): Additional tensformer forward pass arguments
        Returns:
            Tuple[Union[float, Tensor]]: mse loss, last hidden state, (present attention state), 
            (all hidden_states), (attention scores)
        """
        outputs = self.transformer.forward(
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        # If label embeddings are provided, compute loss
        if labels_embeds is not None:
            hidden_states = outputs[0]

            # Flatten the tokens
            loss_fct = nn.MSELoss()
            loss = loss_fct(hidden_states, labels_embeds)

            # loss = loss+ loss_fct(shift_hidden[:,:3], shift_labels[:,:3])
            outputs = (loss,) + (hidden_states, labels_embeds,) + outputs[1:]

        # (loss), last hidden state, (presents), (all hidden_states), (attentions)
        return outputs

    def evaluate(
        self,
        inputs_embeds: Tensor,
        labels_embeds: Tensor,
        **kwargs
    ) -> Tuple[Union[float, Tensor]]:
        """Generate a time-series prediction using the transformer and calc MSE error.
        Args:
            inputs_embeds (Tensor): [B, 1, n_embed] Starting input feature(s)
            labels_embeds (Tensor): [B, T, n_embed] Target output features
            **kwargs (optional): Additional tensformer forward pass arguments
        Returns:
            Tuple[Union[float, Tensor]]: mse loss, last hidden state, (present attention state), 
            (all hidden_states), (attention scores)
        """

        max_length = labels_embeds.size(1)

        outputs = self.transformer.generate(
            inputs_embeds=inputs_embeds,
            max_length=max_length,
            **kwargs
        )
        pred_embeds = outputs[0]

        # Flatten the tokens
        err_fct = nn.MSELoss()
        error = err_fct(pred_embeds, labels_embeds)

        outputs = (error,) + (pred_embeds, labels_embeds,) + outputs[1:]

        return outputs

    def generate(self, *args, **kwargs):
        """
        Generate call is just the forward call of the transformer
        """
        return self.transformer.generate(*args, **kwargs)

    def save_model(self, *args, **kwargs):
        """
        Saves physformer model
        """
        self.transformer.save_model(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        """
        Load a physformer model
        """
        self.transformer.load_model(*args, **kwargs)
