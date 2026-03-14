from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import numpy as np

class AbstractOptimizerMeta(ABC):
    """Базовый класс-плагин для оптимизаторов гиперпараметров"""
    _registry = {}

    def __init_subclass__(cls, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name:
            AbstractOptimizerMeta._registry[name] = cls

    @classmethod
    def get_plugin(cls, name: str):
        plugin_cls = cls._registry.get(name)
        if not plugin_cls:
            raise ValueError(f"Неизвестный оптимизатор: {name}")
        return plugin_cls()

    @abstractmethod
    def optimize(
        self, 
        clusterer_factory,  #функция создания кластеризатора
        data: np.ndarray,
        param_ranges: Dict[str, Tuple[float, float, float]],  # (min, max, delta)
    ) -> Tuple[Dict[str, Any], Tuple[np.ndarray, np.ndarray]]:
        """
        param_ranges: {'c': (2, 6, 1), 'm': (1.5, 2.5, 0.1), ...} например
        Возвращает: (best_params, (centers, labels))
        """
        ...
