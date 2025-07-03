from abc import ABC, abstractmethod
import numpy as np
import pickle

class RnnBase(ABC):
    def __init__(self, input_size, hidden_size, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.reset_parameters()

    @abstractmethod
    def forward(self, inputs: list[np.ndarray], h0: np.ndarray = None) -> list[np.ndarray]:
        """
        Run the forward pass of the RNN/LSTM.
        inputs: list of np.array with shape (1, input_size)
        h0: initial hidden state (1, hidden_size), optional
        returns: list of hidden states at each timestep (list of np.ndarray)
        """
        pass

    @abstractmethod
    def predict(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        """
        Run forward and output predictions.
        """
        pass

    @abstractmethod
    def compute_loss(self, outputs: list[np.ndarray], targets: list[np.ndarray]) -> float:
        """
        Compute the loss (e.g. MSE) for a sequence.
        """
        pass

    @abstractmethod
    def train_step(self, inputs: list[np.ndarray], targets: list[np.ndarray], lr: float = 0.01) -> float:
        """
        Perform one training step and update weights.
        Returns the loss for this step.
        """
        pass

    def evaluate(self, dataset_X: list[list[np.ndarray]], dataset_Y: list[list[np.ndarray]]) -> float:
        """
        Evaluate average loss over dataset.
        """
        total_loss = 0.0
        n = len(dataset_X)
        for x_seq, y_seq in zip(dataset_X, dataset_Y):
            preds = self.predict(x_seq)
            loss = self.compute_loss(preds, y_seq)
            total_loss += loss
        return total_loss / n if n > 0 else 0.0

    def train(self, dataset_X: list[list[np.ndarray]], dataset_Y: list[list[np.ndarray]],
              epochs: int = 1000, lr: float = 0.01, verbose: bool = True, eval_X=None, eval_Y=None):
        """
        Training loop over epochs.
        Optionally evaluate on validation set.
        """
        for epoch in range(epochs):
            total_loss = 0
            for x_seq, y_seq in zip(dataset_X, dataset_Y):
                loss = self.train_step(x_seq, y_seq, lr)
                total_loss += loss
            avg_loss = total_loss / len(dataset_X)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.6f}", end='')
                if eval_X is not None and eval_Y is not None:
                    val_loss = self.evaluate(eval_X, eval_Y)
                    print(f", Val Loss: {val_loss:.6f}")
                else:
                    print()

    @abstractmethod
    def reset_parameters(self):
        """
        Initialize or reset weights.
        """
        pass

    def save(self, filepath: str):
        """
        Save model parameters to a file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, filepath: str):
        """
        Load model parameters from a file.
        """
        with open(filepath, 'rb') as f:
            state_dict = pickle.load(f)
        self.__dict__.update(state_dict)