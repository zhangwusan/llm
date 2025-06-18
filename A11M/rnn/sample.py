import numpy as np
from .base import RnnBase

class RNN(RnnBase):
    """
    A simple vanilla Recurrent Neural Network (RNN) implementation from scratch using NumPy.

    This class processes a sequence of input vectors one step at a time using a basic RNN cell
    with tanh activation and outputs the hidden state at each time step.

    ---
    Benefits:
    - Easy to implement and understand.
    - Suitable for modeling short sequential data (e.g., character-level modeling, simple time series).
    - Fewer parameters than LSTM/GRU, leading to faster computations for small tasks.
    - Good educational tool for learning the inner workings of RNNs.

    Drawbacks:
    - Struggles with long-term dependencies due to vanishing/exploding gradients.
    - No gating mechanism (unlike LSTM/GRU), so memory retention is weak.
    - Cannot handle sequences in parallel — sequential computation is required.
    - Lacks built-in output layer (e.g., softmax for classification).
    """

    def __init__(self, input_size, hidden_size):
        super().__init__(input_size, hidden_size)
        self.W_xh = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1. / hidden_size)
        self.b_h  = np.zeros((1, hidden_size))

    def forward(self, inputs, h0=None):
        """
        Forward pass through the RNN for a given input sequence.

        Parameters:
        - inputs: list of np.ndarray, each with shape (1, input_size)
        - h0: optional initial hidden state of shape (1, hidden_size)

        Returns:
        - List of hidden states at each time step (each of shape (1, hidden_size))
        """
        h = h0 if h0 is not None else np.zeros((1, self.hidden_size))
        hidden_states = []

        for x in inputs:
            h: np.ndarray = np.tanh(x @ self.W_xh + h @ self.W_hh + self.b_h)
            hidden_states.append(h.copy())

        return hidden_states