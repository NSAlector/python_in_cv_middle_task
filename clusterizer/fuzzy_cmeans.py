import numpy as np
from .interface import Clusterizer

class FuzzyCMeansClusterizer(Clusterizer):
    """
    Реализация алгоритма нечётких C-средних (Fuzzy C-means) с L1-расстоянием.
    """

    def __init__(self, c: int, m: float = 2.0, eps: float = 1e-4, max_iter: int = 100, seed: int = None):
        """
        Параметры
        ----------
        c : int – количество кластеров
        m : float – степень размытости (fuzziness), по умолчанию 2.0
        eps : float – порог сходимости (максимальное изменение принадлежностей)
        max_iter : int – максимальное число итераций
        seed : int – для воспроизводимости случайной инициализации
        """
        self.c = c
        self.m = m
        self.eps = eps
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, data: np.ndarray):
        """
        Запускает кластеризацию методом нечётких C-средних.

        Возвращает
        ----------
        centers : np.ndarray, форма (c, n_features)
        memberships : np.ndarray, форма (n_samples, c)
        """
        n_samples, n_features = data.shape

        # Генератор случайных чисел
        rng = np.random.default_rng(self.seed)

        # 1. Инициализация матрицы принадлежностей случайными значениями из [0,1]
        #    с нормировкой по строкам (сумма по кластерам = 1)
        memberships = rng.random((n_samples, self.c))
        memberships = memberships / memberships.sum(axis=1, keepdims=True)

        # 2. Итерации до сходимости
        for iteration in range(self.max_iter):
            # Сохраняем старую матрицу для проверки сходимости
            old_memberships = memberships.copy()

            # ---- Вычисление центров кластеров ----
            # numerator = sum (u^m * x), denominator = sum (u^m)
            um = memberships ** self.m                     # shape (n_samples, c)
            # центры: (c, n_features)
            centers = (um.T @ data) / um.sum(axis=0)[:, np.newaxis]

            # ---- Вычисление расстояний от каждой точки до каждого центра (L1) ----
            # L1: |x - center| поэлементно, затем сумма по признакам
            # data: (n_samples, n_features) -> (n_samples, 1, n_features)
            # centers: (c, n_features) -> (1, c, n_features)
            # разность: (n_samples, c, n_features)
            diffs = np.abs(data[:, np.newaxis, :] - centers[np.newaxis, :, :])
            distances = np.sum(diffs, axis=2)              # shape (n_samples, c)

            # Обработка нулевых расстояний: если точка совпадает с центром,
            # то её принадлежность этому кластеру должна быть 1, остальным 0.
            # Для численной устойчивости заменим нули на очень маленькое число
            distances = np.where(distances < 1e-12, 1e-12, distances)

            # ---- Пересчёт принадлежностей ----
            # Формула: u_ik = 1 / ( sum_{j=1..c} (d_ik / d_ij)^(2/(m-1)) )
            # Сначала вычислим (d_ik)^(2/(m-1)) для всех i,k
            exponent = 2.0 / (self.m - 1)
            # distances: (n_samples, c) -> возводим в степень
            d_pow = distances ** exponent                  # shape (n_samples, c)

            # Сумма по кластерам (j) для каждой точки
            sum_d_pow = np.sum(d_pow, axis=1, keepdims=True)  # (n_samples, 1)

            # Тогда принадлежности: d_pow / sum_d_pow? Проверим:
            # u_ik = 1 / ( sum_j (d_ik/d_ij)^exponent ) = 1 / ( d_ik^exponent * sum_j (1/d_ij^exponent) )
            # = (1 / d_ik^exponent) / ( sum_j (1/d_ij^exponent) )
            # Это не совпадает с d_pow / sum_d_pow. На самом деле формулу можно упростить:
            # Стандартная формула: u_ik = 1 / sum_j ( (d_ik/d_ij)^(2/(m-1)) )
            # Перепишем: u_ik = 1 / ( d_ik^(2/(m-1)) * sum_j (1 / d_ij^(2/(m-1)) ) )
            # = (1 / d_ik^(2/(m-1))) / sum_j (1 / d_ij^(2/(m-1)))
            # Обозначим inv_d_pow = 1 / d_pow. Тогда u_ik = inv_d_pow[i,k] / sum_j inv_d_pow[i,j].
            # Это эквивалентно: inv_d_pow = distances ** (-exponent) = 1 / d_pow.
            # Тогда принадлежности = inv_d_pow / inv_d_pow.sum(axis=1, keepdims=True)
            inv_d_pow = 1.0 / d_pow
            memberships = inv_d_pow / inv_d_pow.sum(axis=1, keepdims=True)

            # Проверка сходимости (максимальное изменение элементов матрицы)
            max_change = np.max(np.abs(memberships - old_memberships))
            if max_change < self.eps:
                break

        # Возвращаем центры и финальные принадлежности
        return centers, memberships