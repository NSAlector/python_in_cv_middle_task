import numpy as np
import itertools
from .interface import Optimizer

class GridSearchOptimizer(Optimizer):
    """
    Оптимизатор методом полного перебора по равномерной сетке значений параметров.
    Критерий: минимизация суммы взвешенных расстояний (целевая функция FCM)
    с предпочтением меньшего числа кластеров при равных значениях.
    """

    def __init__(self, verbose=False):
        """
        verbose : bool – выводить информацию о текущей комбинации
        """
        self.verbose = verbose

    def _generate_grid(self, param_grid):
        """
        Генерирует все комбинации параметров из param_grid.
        Возвращает итератор словарей.
        """
        grid = {}
        for param_name, spec in param_grid.items():
            if isinstance(spec, (tuple, list)) and len(spec) == 3:
                # Это диапазон (min, max, step)
                min_val, max_val, step = spec
                if isinstance(min_val, int) and isinstance(step, int):
                    # Целочисленный диапазон
                    values = list(range(min_val, max_val + 1, step))
                else:
                    # Вещественный диапазон: используем np.arange с учётом точности
                    # Чтобы включить max_val, добавляем небольшой допуск
                    values = []
                    val = min_val
                    while val <= max_val + 1e-9:
                        values.append(val)
                        val += step
            else:
                # Уже готовый список значений
                values = spec
            grid[param_name] = values

        # Декартово произведение
        keys = grid.keys()
        for combination in itertools.product(*grid.values()):
            yield dict(zip(keys, combination))

    def optimize(self, data, clusterizer_class, param_grid):
        """
        Выполняет полный перебор по сетке и возвращает лучший результат.
        """
        best_score = float('inf')
        best_c = None
        best_params = None
        best_result = None

        for params in self._generate_grid(param_grid):
            # Фиксируем seed для воспроизводимости, если не задан
            if 'seed' not in params:
                params['seed'] = 42

            if self.verbose:
                print(f"Пробуем параметры: {params}")

            # Создаём экземпляр кластеризатора
            try:
                clusterizer = clusterizer_class(**params)
            except TypeError as e:
                if self.verbose:
                    print(f"Ошибка создания кластеризатора: {e}")
                continue

            # Обучаем
            centers, memberships = clusterizer.fit(data)

            # Получаем значение степени размытости (m) из объекта
            m = getattr(clusterizer, 'm', 2.0)

            # Вычисляем сумму взвешенных расстояний (целевая функция J)
            # Расстояние L1 от каждой точки до каждого центра
            # data: (n_samples, n_features)
            # centers: (n_clusters, n_features)
            diffs = np.abs(data[:, np.newaxis, :] - centers[np.newaxis, :, :])
            distances = np.sum(diffs, axis=2)                     # (n_samples, n_clusters)

            # Взвешивание на u^m
            u_m = memberships ** m
            score = np.sum(u_m * distances)

            # Количество кластеров
            c = centers.shape[0]

            # Сравнение: сначала по score, затем по c (меньше лучше)
            if (score < best_score) or (np.isclose(score, best_score) and c < best_c):
                best_score = score
                best_c = c
                best_params = params
                best_result = (centers, memberships)

        return best_params, best_result, best_score