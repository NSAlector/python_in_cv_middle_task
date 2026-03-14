from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    """
    Абстрактный базовый класс для оптимизаторов гиперпараметров.
    """

    @abstractmethod
    def optimize(self, data: np.ndarray, clusterizer_class, param_grid: dict):
        """
        Подбирает оптимальные гиперпараметры для заданного класса кластеризатора.

        Параметры
        ----------
        data : np.ndarray, форма (n_samples, n_features) – данные для кластеризации
        clusterizer_class : class – класс, реализующий интерфейс Clusterizer
        param_grid : dict – словарь с параметрами для перебора.
                     Ключи – имена параметров конструктора clusterizer_class.
                     Значения могут быть:
                        - списком значений [val1, val2, ...]
                        - кортежем (min, max, step) для равномерной сетки

        Возвращает
        ----------
        tuple (best_params, best_result, best_score)
            best_params : dict – словарь лучших параметров
            best_result : tuple (centers, memberships) – результат кластеризации с лучшими параметрами
            best_score : float – значение критерия для лучшего решения
        """
        pass