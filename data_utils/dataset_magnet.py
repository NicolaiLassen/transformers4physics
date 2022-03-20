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
import time
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from data_utils.embedding_data_handler import EmbeddingDataHandler

logger = logging.getLogger(__name__)

class MagnetDataHandler(EmbeddingDataHandler):
    """Built in embedding data handler for Lorenz system
    """
    class MagnetDataset(Dataset):
        """Dataset for training Lorenz embedding model.
        Args:
            examples (List): list of training/testing examples
        """
        def __init__(self, examples: List):
            """Constructor
            """
            self.examples = examples

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {"states": self.examples[i]}

    @dataclass
    class MagnetDataCollator:
        """Data collator for lorenz embedding problem
        """
        # Default collator
        def __call__(self, examples:List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            # Stack examples in mini-batch
            x_data_tensor =  torch.stack([example['states'] for example in examples])

            return {"states": x_data_tensor}

    def createTrainingLoader(self, 
        file_path: str,  #hdf5 file
        block_size: int, # Length of time-series
        stride: int = 1,
        ndata: int = -1,
        batch_size: int = 32,
        shuffle: bool = True,
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ) -> DataLoader:
        """Creating training data loader for Lorenz system.
        For a single training simulation, the total time-series is sub-chunked into
        smaller blocks for training.
        Args:
            file_path (str): Path to HDF5 file with training data
            block_size (int): The length of time-series blocks
            stride (int): Stride of each time-series block
            ndata (int, optional): Number of training time-series. If negative, all of the provided 
            data will be used. Defaults to -1.
            batch_size (int, optional): Training batch size. Defaults to 32.
            shuffle (bool, optional): Turn on mini-batch shuffling in dataloader. Defaults to True.
        Returns:
            (DataLoader): Training loader
        """
        logger.info('Creating training loader.')
        assert os.path.isfile(file_path), "Training HDF5 file {} not found".format(file_path)
        
        examples = []
        with h5py.File(file_path, "r") as f:
            # Iterate through stored time-series
            samples = 0
            for key in f.keys():
                data_series = torch.Tensor(np.array(f[key]))
                # Stride over time-series
                for i in range(0,  data_series.size(0) - block_size + 1, stride):  # Truncate in block of block_size
                    examples.append(data_series[i : i + block_size].unsqueeze(0))

                samples = samples + 1
                if(ndata > 0 and samples > ndata): #If we have enough time-series samples break loop
                    break

        # Calculate normalization constants
        data = torch.cat(examples, dim=0)
        self.mu = torch.tensor([torch.mean(data[:,:,0]), torch.mean(data[:,:,1]), torch.mean(data[:,:,2])])
        self.std = torch.tensor([torch.std(data[:,:,0]), torch.std(data[:,:,1]), torch.std(data[:,:,2])])

        # Needs to min-max normalization due to the reservoir matrix, needing to have a spectral density below 1
        if(data.size(0) < batch_size):
            logger.warning('Lower batch-size to {:d}'.format(data.size(0)))
            batch_size = data.size(0)

        # Create dataset, collator, and dataloader
        dataset = self.MagnetDataset(data.to(device))
        data_collator = self.MagnetDataCollator()
        training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, drop_last=True)
        
        return training_loader

    def createTestingLoader(self, 
        file_path: str,
        block_size: int,
        ndata: int = -1,
        batch_size: int =32,
        shuffle: bool =False
    ) -> DataLoader:
        """Creating testing/validation data loader for Lorenz system.
        For a data case with time-steps [0,T], this method extract a smaller
        time-series to be used for testing [0, S], s.t. S < T.
        Args:
            file_path (str): Path to HDF5 file with testing data
            block_size (int): The length of testing time-series
            ndata (int, optional): Number of testing time-series. If negative, all of the provided 
            data will be used. Defaults to -1.
            batch_size (int, optional): Testing batch size. Defaults to 32.
            shuffle (bool, optional): Turn on mini-batch shuffling in dataloader. Defaults to False.
        Returns:
            (DataLoader): Testing/validation data loader
        """
        logger.info('Creating testing loader')
        assert os.path.isfile(file_path), "Eval HDF5 file {} not found".format(file_path)

        examples = []
        with h5py.File(file_path, "r") as f:
            # Iterate through stored time-series
            samples = 0
            for key in f.keys():
                data_series = torch.Tensor(np.array(f[key]))
                # Stride over time-series
                for i in range(0,  data_series.size(0) - block_size + 1, block_size):  # Truncate in block of block_size
                    examples.append(data_series[i : i + block_size].unsqueeze(0))
                    break

                samples = samples + 1
                if(ndata > 0 and samples >= ndata): #If we have enough time-series samples break loop
                    break

        # Combine data-series
        data = torch.cat(examples, dim=0)
        if(data.size(0) < batch_size):
            logger.warning('Lower batch-size to {:d}'.format(data.size(0)))
            batch_size = data.size(0)

        dataset = self.MagnetDataset(data)
        data_collator = self.MagnetDataCollator()
        testing_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, drop_last=False)

        return testing_loader

class MicroMagnetismDataset(PhysicalDataset):
    """Dataset for the 
    """

    def embed_data(self, h5_file: h5py.File, embedder: EmbeddingModel) -> None:
        """Embeds micro magnet data into a 1D vector representation for the transformer.
        Args:
            h5_file (h5py.File): HDF5 file object of raw data
            embedder (EmbeddingModel): Embedding neural network
        """
        
        # Iterate through stored time-series
        samples = 0
        for key in h5_file.keys():
            data_series = torch.Tensor(h5_file[key]).to(embedder.devices[0])
            with torch.no_grad():
                embedded_series = embedder.embed(data_series).cpu()

            # Stride over time-series
            for i in range(0, data_series.size(0) - self.block_size + 1,
                           self.stride):  # Truncate in block of block_size

                data_series0 = embedded_series[i: i + self.block_size]
                self.examples.append(data_series0)

                if self.eval:
                    self.states.append(
                        data_series[i: i + self.block_size].cpu())

            samples = samples + 1
            # If we have enough time-series samples break loop
            if (self.ndata > 0 and samples >= self.ndata):
                break

        logger.info(
            'Collected {:d} time-series from hdf5 file. Total of {:d} time-series.'.format(
                samples, len(self.examples))
        )
