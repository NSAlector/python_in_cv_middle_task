import numpy as np
import itertools


class GridSearchOptimizer:
    def __init__(self, param_grid, delta=None):
        """
        Args:
            param_grid: словарь с параметрами и их возможными значениями
            delta: словарь с шагами для каждого параметра
        """
        self.param_grid = param_grid
        self.delta = delta or {}
        self.best_params = None
        self.best_score = float('inf')
        self.best_result = None

    def _generate_param_combinations(self):
        """
        Генерирует все комбинации параметров для перебора
        """
        expanded_grid = {}

        for param_name, param_values in self.param_grid.items():
            if isinstance(param_values, (list, tuple)):
                if len(param_values) == 2 and param_name in self.delta:
                    min_val, max_val = param_values
                    step = self.delta[param_name]

                    # Проверяем направление диапазона
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val

                    values = []
                    val = min_val

                    # Для целых чисел
                    if isinstance(min_val, int) and step == int(step):
                        while val <= max_val + 1e-10:
                            values.append(int(round(val)))
                            val += step
                    else:
                        # Для вещественных чисел
                        while val <= max_val + 1e-10:
                            values.append(round(val, 10))
                            val += step

                    if values and abs(values[-1] - max_val) > 1e-10:
                        values.append(max_val if isinstance(max_val, int) else round(max_val, 10))

                    expanded_grid[param_name] = list(set(values))
                    expanded_grid[param_name].sort()
                    print(
                        f"Параметр {param_name}: диапазон {min_val}-{max_val} с шагом {step} "
                        f"-> {len(expanded_grid[param_name])} значений: {expanded_grid[param_name]}")
                else:
                    expanded_grid[param_name] = list(param_values)
                    print(f"Параметр {param_name}: фиксированные значения {param_values}")
            else:
                expanded_grid[param_name] = [param_values]
                print(f"Параметр {param_name}: одно значение {param_values}")

        if not expanded_grid:
            print("expanded_grid пуст!")
            return []

        param_names = list(expanded_grid.keys())
        param_values = list(expanded_grid.values())

        combinations = []
        for values in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, values)))

        print(f"Сгенерировано {len(combinations)} комбинаций параметров")
        return combinations

    def _evaluate_clustering(self, clusterizer_class, x, params):
        """
        Оценивает качество кластеризации
        """
        try:
            if params.get('n_clusters', 0) > len(x):
                return float('inf'), None

            clusterizer = clusterizer_class(**params)
            clusterizer.fit(x)

            labels = clusterizer.get_labels()
            centers = clusterizer.get_centers()

            if labels is None or centers is None:
                return float('inf'), None

            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)

            # 1. Сумма разброса значений в кластере
            total_within_cluster_distance = 0
            for i in range(n_clusters):
                cluster_points = x[labels == i]
                if len(cluster_points) > 0:
                    distances = np.sum(np.abs(cluster_points - centers[i]), axis=1)
                    total_within_cluster_distance += np.sum(distances)

            # 2. Нормализуем на количество точек
            avg_within_distance = total_within_cluster_distance / len(x)

            # 3. Критерий: разброс * количество кластеров
            score = avg_within_distance * n_clusters

            return score, clusterizer

        except Exception as e:
            print(f"Ошибка при оценке кластеризации: {e}")
            return float('inf'), None

    def optimize(self, clusterizer_class, x):
        """
        Выполняет оптимизацию гиперпараметров
        """
        print("\n" + "=" * 50)
        print("ЗАПУСК ОПТИМИЗАЦИИ ГИПЕРПАРАМЕТРОВ")
        print("=" * 50)

        if x is None or len(x) == 0:
            raise ValueError("Данные для кластеризации пусты")

        print(f"Данные: {len(x)} точек, {x.shape[1]} признаков")

        param_combinations = self._generate_param_combinations()

        if not param_combinations:
            print(f"param_grid: {self.param_grid}")
            print(f"delta: {self.delta}")
            raise ValueError("Не сгенерировано ни одной комбинации параметров")

        print(f"\nПеребор {len(param_combinations)} комбинаций:")

        successful_combinations = 0

        for i, params in enumerate(param_combinations):
            print(f"\n  Комбинация {i + 1}/{len(param_combinations)}: {params}")

            score, clusterizer = self._evaluate_clustering(clusterizer_class, x, params)

            if score < float('inf'):
                successful_combinations += 1
                print(f"    Оценка: {score:.4f}")

                if score < self.best_score:
                    self.best_score = score
                    self.best_params = params
                    self.best_result = clusterizer
                    print(f"    Новая лучшая оценка!")
            else:
                print(f"    Оценка не удалась")

        if self.best_result is None:
            raise ValueError(f"Не найдено успешных комбинаций параметров из {len(param_combinations)}")

        print("\n" + "=" * 50)
        print(f"ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        print(f"Лучшие параметры: {self.best_params}")
        print(f"Лучшая оценка: {self.best_score:.4f}")
        print(f"Успешных комбинаций: {successful_combinations}/{len(param_combinations)}")
        print("=" * 50)

        return self.best_params, self.best_result

    def get_best_params(self):
        """Возвращает лучшие параметры"""
        return self.best_params

    def get_best_score(self):
        """Возвращает лучшую оценку"""
        return self.best_score


def get_optimizer():
    print("get_optimizer() вызван в optimizer.py")
    return GridSearchOptimizer
