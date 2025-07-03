import numpy as np
from .base import BaseScaler


class StandardScaler(BaseScaler):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        self.mean = np.mean(data)
        self.std = np.std(data)
        if self.std == 0:
            raise ValueError("Standard deviation is zero — cannot scale.")

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean