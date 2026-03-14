from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np

class Clusterizer(ABC):
    @abstractmethod
    def fit(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(centers, labels)"""
        pass

class HyperparamOptimizerInterface(ABC):
    @abstractmethod
    def grid_search(self, param_ranges: Dict[str, tuple]) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        pass
