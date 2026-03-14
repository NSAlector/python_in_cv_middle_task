import numpy as np


class FuzzyCMeans:
    def __init__(self, n_clusters=3, m=2.0, eps=1e-4, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.eps = eps
        self.max_iter = max_iter
        self.random_state = random_state
        self.centers = None
        self.membership = None
        self.labels = None

    def fit(self, matrix):
        """
        Args:
            matrix: матрица признаков (n_samples, n_features)

        Returns:
            self
        """
        print(f"FuzzyCMeans.fit() вызван с X.shape={matrix.shape if matrix is not None else 'None'}")
        print(f"Параметры: n_clusters={self.n_clusters}, m={self.m}, eps={self.eps}")

        if matrix is None:
            raise ValueError("X is None")

        if len(matrix) == 0:
            raise ValueError("X is empty")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = matrix.shape

        if self.n_clusters > n_samples:
            print(f"Предупреждение: кластеров ({self.n_clusters}) больше чем точек ({n_samples})")
            self.n_clusters = n_samples

        # 1: Случайная инициализация принадлежности [0,1]
        self.membership = np.random.rand(n_samples, self.n_clusters)
        self.membership = self.membership / self.membership.sum(axis=1, keepdims=True)

        n_iter = 0

        while n_iter < self.max_iter:
            # 2: Вычисление центроидов
            self.centers = self._compute_centers(matrix)

            # 3: Обновление матрицы принадлежности
            new_membership = self._update_membership(matrix)

            # 4: Проверка сходимости
            diff = np.abs(new_membership - self.membership).max()

            self.membership = new_membership
            n_iter += 1

            if diff < self.eps:
                break

        # Определение меток кластеров
        self.labels = np.argmax(self.membership, axis=1)

        print(f"FCM сошелся за {n_iter} итераций")
        print(f"Распределение по кластерам: {np.bincount(self.labels)}")
        return self

    def _compute_centers(self, x):
        """
        Вычисление центроидов кластеров
        """
        n_features = x.shape[1]
        centers = np.zeros((self.n_clusters, n_features))

        membership_power = self.membership ** self.m

        for j in range(self.n_clusters):
            weighted_sum = np.sum(membership_power[:, j:j + 1] * x, axis=0)

            denominator = np.sum(membership_power[:, j])

            if denominator > 1e-10:
                centers[j] = weighted_sum / denominator
            else:
                random_idx = np.random.randint(len(x))
                centers[j] = x[random_idx].copy()

        return centers

    def _update_membership(self, x):
        """
        Обновление матрицы принадлежности
        """
        n_samples = x.shape[0]
        new_membership = np.zeros((n_samples, self.n_clusters))

        distances = np.zeros((n_samples, self.n_clusters))
        for j in range(self.n_clusters):
            distances[:, j] = np.sum(np.abs(x - self.centers[j]), axis=1)

        distances = np.maximum(distances, 1e-10)

        exponent = 2.0 / (self.m - 1) if self.m > 1 else 1.0

        for i in range(n_samples):
            for j in range(self.n_clusters):
                denominator = np.sum((distances[i, j] / distances[i, :]) ** exponent)
                if denominator > 0:
                    new_membership[i, j] = 1.0 / denominator
                else:
                    new_membership[i, j] = 1.0 / self.n_clusters

        return new_membership

    def predict(self, x):
        """Предсказание кластеров для новых данных"""
        if self.centers is None:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")

        distances = np.zeros((x.shape[0], self.n_clusters))
        for j in range(self.n_clusters):
            distances[:, j] = np.sum(np.abs(x - self.centers[j]), axis=1)

        return np.argmin(distances, axis=1)

    def get_centers(self):
        """Возвращает центры кластеров"""
        return self.centers

    def get_membership(self):
        """Возвращает матрицу принадлежности"""
        return self.membership

    def get_labels(self):
        """Возвращает метки кластеров"""
        return self.labels


def get_clusterizer():
    return FuzzyCMeans
