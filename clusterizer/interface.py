from abc import ABC, abstractmethod
import numpy as np

class Clusterizer(ABC):
    """
    Абстрактный базовый класс для всех алгоритмов кластеризации.
    """

    @abstractmethod
    def fit(self, data: np.ndarray):
        """
        Обучает модель кластеризации на данных data.

        Параметры
        ----------
        data : np.ndarray, форма (n_samples, n_features)

        Возвращает
        ----------
        tuple (centers, memberships)
            centers : np.ndarray, форма (n_clusters, n_features) – центры кластеров
            memberships : np.ndarray, форма (n_samples, n_clusters) – матрица принадлежностей
                          (для чётких алгоритмов – бинарная, для нечётких – вещественная)
        """
        pass