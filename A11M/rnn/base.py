from abc import ABC, abstractmethod
import numpy as np


class RnnBase(ABC):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
    
    @abstractmethod
    def forward(self, inputs: list[np.ndarray], h0: np.ndarray = None) -> list[np.ndarray]:
        """
        inputs: list of np.array with shape (1, input_size)
        h0: initial hidden state (1, hidden_size)
        returns: list of hidden states at each timestep
        """
        pass

    def reset_parameters(self):
        """
        Optional: Subclasses can override this if they want custom weight initialization.
        """
        pass
