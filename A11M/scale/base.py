from abc import ABC, abstractmethod
import numpy as np

class BaseScaler(ABC):
    @abstractmethod
    def fit(self, data: np.ndarray):
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        pass

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)