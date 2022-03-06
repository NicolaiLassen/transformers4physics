"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import torch
from typing import Dict, List
from dataclasses import dataclass

Tensor = torch.Tensor

@dataclass
class DataCollator:
    """
    Data collator used for training datasets. 
    Combines examples in a minibatch into one tensor.
    
    Args:
        examples (List[Dict[str, Tensor]]): List of training examples. An example
            should be a dictionary of tensors from the dataset.
        Returns:
            Dict[str, Tensor]: Minibatch dictionary of combined example data tensors
    """
    # Default collator
    def __call__(self, examples:List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        mini_batch = {}
        for key in examples[0].keys():
            mini_batch[key] = self._tensorize_batch([example[key] for example in examples])

        return mini_batch

    def _tensorize_batch(self, examples: List[Tensor]) -> Tensor:
        if not torch.is_tensor(examples[0]):
            return examples

        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)

        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            raise ValueError("Padding not currently supported for physics transformers")
            return

@dataclass
class EvalDataCollator:
    """
    Data collator used for evaluation/testing datasets.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    # Default collator
    def __call__(self, examples:List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        eval_data = {}
        for key in examples[0].keys():
            eval_data[key] = self._tensorize_batch([example[key] for example in examples])
        # inputs = self._tensorize_batch([example['input'] for example in examples])
        # props = self._tensorize_batch([example['positions'] for example in examples])
        # targets = self._tensorize_batch([example['targets'] for example in examples])

        return eval_data

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)

        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            raise ValueError("Padding not currently supported for physics transformers")
            return