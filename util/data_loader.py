
import os
from tkinter.messagebox import NO
from typing import Tuple

import h5py
import numpy as np
import torch
from embedding.embedding_model import EmbeddingModel
from torch.utils.data.dataset import Dataset

Tensor = torch.Tensor

class MagDataset(Dataset):
    def __init__(self, examples, fields, embedded=None):
        self.examples = examples
        self.fields = fields
        self.embedded =  embedded if embedded is not None else torch.zeros((examples.size(0))) 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return {"states": self.examples[i], "fields": self.fields[i], "embedded": self.embedded[i]}

def read_and_embbed_h5_dataset(
    file_path: str,
    embedder: EmbeddingModel,
    block_size: int,
    batch_size: int = 32,
    stride: int = 5,
    n_data: int = -1,
) -> Tensor:
    assert os.path.isfile(
        file_path), "Training HDF5 file {} not found".format(file_path)

    seq = []
    fields = []
    embedded_seq = []
    with h5py.File(file_path, "r") as f:

        n_seq = 0
        for key in f.keys():
            data_series = torch.Tensor(np.array(f[key]['sequence']))
            field = torch.Tensor(np.array(f[key]['field'][:2])).unsqueeze(0)
            
            with torch.no_grad():
                embedded_series = embedder.embed(data_series, field).cpu()

            # Truncate in block of block_size
            for i in range(0,  data_series.size(0) - block_size + 1, stride):
                seq.append(data_series[i: i + block_size].unsqueeze(0))
                fields.append(field)
                embedded_seq.append(embedded_series[i: i + block_size].unsqueeze(0))

            n_seq = n_seq + 1
            if(n_data > 0 and n_seq >= n_data):  # If we have enough time-series samples break loop
                break

    seq_tensor = torch.cat(seq,dim=0)
    fields_tensor = torch.cat(fields,dim=0)
    embedded_tensor = torch.cat(embedded_seq,dim=0)
    data = MagDataset(seq_tensor, fields_tensor, embedded_tensor)

    if seq_tensor.size(0) < batch_size:
        batch_size = seq_tensor.size(0)

    return data

def read_h5_dataset(
    file_path: str,
    block_size: int,
    batch_size: int = 32,
    stride: int = 5,
    n_data: int = -1,
) -> Tensor:

    assert os.path.isfile(
        file_path), "Training HDF5 file {} not found".format(file_path)

    seq = []
    fields = []
    with h5py.File(file_path, "r") as f:

        n_seq = 0
        for key in f.keys():
            data_series = torch.Tensor(np.array(f[key]['sequence']))
            # Truncate in block of block_size
            for i in range(0,  data_series.size(0) - block_size + 1, stride):
                seq.append(data_series[i: i + block_size].unsqueeze(0))
                fields.append(torch.Tensor(np.array(f[key]['field'][:2])).unsqueeze(0))

            n_seq = n_seq + 1
            if(n_data > 0 and n_seq >= n_data):  # If we have enough time-series samples break loop
                break

    seq_tensor = torch.cat(seq,dim=0)
    fields_tensor = torch.cat(fields,dim=0)
    data = MagDataset(seq_tensor, fields_tensor)
    if seq_tensor.size(0) < batch_size:
        batch_size = seq_tensor.size(0)

    return data