from abc import abstractmethod
from ast import Tuple


class EmbeddingDataHandler(object):
    """Base class for embedding data handlers.
    Data handlers are used to create the training and
    testing datasets.
    """
    mu = None
    std = None

    @property
    def norm_params(self) -> Tuple:
        """Get normalization parameters
        Raises:
            ValueError: If normalization parameters have not been initialized
        Returns:
            (Tuple): mean and standard deviation
        """
        if self.mu is None or self.std is None:
            raise ValueError("Normalization constants set yet!")
        return self.mu, self.std

    @abstractmethod
    def createTrainingLoader(self, *args, **kwargs):
        pass

    @abstractmethod
    def createTestingLoader(self, *args, **kwargs):
        pass
