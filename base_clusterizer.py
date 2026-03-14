from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class AbstractClustererMeta(ABC):
    """Базовый класс-плагин для кластеризаторов"""
    _registry = {}

    def __init_subclass__(cls, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name:
            AbstractClustererMeta._registry[name] = cls

    @classmethod
    def get_plugin(cls, name: str):
        plugin_cls = cls._registry.get(name)
        if not plugin_cls:
            raise ValueError(f"Неизвестный кластеризатор: {name}")
        return plugin_cls()

    @abstractmethod
    def fit(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """data -> (centers, labels)"""
        ...
