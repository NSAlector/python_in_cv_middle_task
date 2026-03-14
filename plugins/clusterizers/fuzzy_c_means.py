from __future__ import annotations

import random
from typing import Sequence


Number = int | float


class FuzzyCMeansClusterizer:
    """Fuzzy C-Means with L1 distance."""

    def __init__(
        self,
        c: int,
        m: float = 2.0,
        eps: float = 1e-4,
        max_iter: int = 100,
        seed: int = 42,
    ) -> None:
        if c <= 0:
            raise ValueError("Parameter c must be positive.")
        if m <= 1.0:
            raise ValueError("Parameter m must be greater than 1.")
        if eps <= 0:
            raise ValueError("Parameter eps must be positive.")
        if max_iter <= 0:
            raise ValueError("Parameter max_iter must be positive.")

        self.c = c
        self.m = m
        self.eps = eps
        self.max_iter = max_iter
        self.seed = seed

    def fit_predict(self, data: Sequence[Sequence[Number] | Number]) -> dict[str, object]:
        vectors = [_to_vector(point) for point in data]
        if not vectors:
            raise ValueError("Input data is empty.")
        dim = len(vectors[0])
        if any(len(vector) != dim for vector in vectors):
            raise ValueError("All data points must have the same dimensionality.")

        n = len(vectors)
        memberships = self._initialize_memberships(n)
        centroids = self._compute_centroids(vectors, memberships, dim)

        for _ in range(self.max_iter):
            previous = [row[:] for row in memberships]
            centroids = self._compute_centroids(vectors, memberships, dim)
            memberships = self._update_memberships(vectors, centroids)

            max_change = max(
                abs(memberships[i][j] - previous[i][j]) for i in range(n) for j in range(self.c)
            )
            if max_change <= self.eps:
                break

        labels = [max(range(self.c), key=lambda idx: memberships[i][idx]) for i in range(n)]
        return {
            "centroids": centroids,
            "memberships": memberships,
            "labels": labels,
        }

    def _initialize_memberships(self, n: int) -> list[list[float]]:
        rng = random.Random(self.seed)
        memberships: list[list[float]] = []
        for _ in range(n):
            values = [rng.random() + 1e-12 for _ in range(self.c)]
            total = sum(values)
            memberships.append([value / total for value in values])
        return memberships

    def _compute_centroids(
        self,
        vectors: list[list[float]],
        memberships: list[list[float]],
        dim: int,
    ) -> list[list[float]]:
        centroids = [[0.0] * dim for _ in range(self.c)]
        denominators = [0.0] * self.c

        for i, vector in enumerate(vectors):
            for j in range(self.c):
                weight = memberships[i][j] ** self.m
                denominators[j] += weight
                for d in range(dim):
                    centroids[j][d] += weight * vector[d]

        for j in range(self.c):
            if denominators[j] == 0:
                continue
            inv = 1.0 / denominators[j]
            centroids[j] = [value * inv for value in centroids[j]]
        return centroids

    def _update_memberships(
        self,
        vectors: list[list[float]],
        centroids: list[list[float]],
    ) -> list[list[float]]:
        exponent = 2.0 / (self.m - 1.0)
        memberships: list[list[float]] = []

        for vector in vectors:
            distances = [_l1_distance(vector, centroid) for centroid in centroids]
            zeros = [idx for idx, dist in enumerate(distances) if dist == 0.0]
            if zeros:
                share = 1.0 / len(zeros)
                memberships.append([share if idx in zeros else 0.0 for idx in range(self.c)])
                continue

            row = []
            for j in range(self.c):
                numerator = distances[j]
                denominator = sum((numerator / dist) ** exponent for dist in distances)
                row.append(1.0 / denominator)
            memberships.append(row)
        return memberships


def _to_vector(point: Sequence[Number] | Number) -> list[float]:
    if isinstance(point, (int, float)):
        return [float(point)]
    return [float(value) for value in point]


def _l1_distance(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))
