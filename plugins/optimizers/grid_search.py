from __future__ import annotations

from itertools import product
from typing import Any, Sequence


Number = int | float


class GridSearchHyperparameterOptimizer:
    """Uniform grid search for clusterizer hyperparameters."""

    def optimize(
        self,
        clusterizer_cls: type,
        data: Sequence[Sequence[Number] | Number],
        param_ranges: dict[str, tuple[Number, Number]],
        deltas: dict[str, Number],
        fixed_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not param_ranges:
            raise ValueError("param_ranges cannot be empty.")

        fixed_params = fixed_params or {}
        grid = self._build_grid(param_ranges, deltas)
        if not grid:
            raise ValueError("No hyperparameter combinations generated from provided ranges.")

        best_result: dict[str, Any] | None = None
        best_params: dict[str, Any] | None = None
        best_key: tuple[float, int] | None = None

        for params in grid:
            clusterizer = clusterizer_cls(**fixed_params, **params)
            result = clusterizer.fit_predict(data)
            spread = self._cluster_spread(data, result["labels"], result["centroids"])
            clusters_used = len(set(result["labels"]))

            key = (spread, clusters_used)
            if best_key is None or key < best_key:
                best_key = key
                best_result = result
                best_params = params

        if best_result is None or best_params is None or best_key is None:
            raise RuntimeError("Grid search failed to evaluate candidates.")

        return {
            "best_params": best_params,
            "best_score": best_key[0],
            "clusters_used": best_key[1],
            "result": best_result,
        }

    def _build_grid(
        self,
        param_ranges: dict[str, tuple[Number, Number]],
        deltas: dict[str, Number],
    ) -> list[dict[str, Number]]:
        keys = list(param_ranges.keys())
        value_lists: list[list[Number]] = []

        for key in keys:
            if key not in deltas:
                raise ValueError(f"Missing delta for parameter '{key}'.")
            start, end = param_ranges[key]
            step = deltas[key]
            values = _uniform_values(start, end, step)
            if not values:
                raise ValueError(f"No grid values generated for parameter '{key}'.")
            value_lists.append(values)

        combinations = []
        for values in product(*value_lists):
            combinations.append(dict(zip(keys, values)))
        return combinations

    @staticmethod
    def _cluster_spread(
        data: Sequence[Sequence[Number] | Number],
        labels: Sequence[int],
        centroids: Sequence[Sequence[float]],
    ) -> float:
        vectors = [_to_vector(point) for point in data]
        total = 0.0
        for vector, label in zip(vectors, labels):
            total += _l1_distance(vector, centroids[label])
        return total


def _uniform_values(start: Number, end: Number, step: Number) -> list[Number]:
    if step <= 0:
        raise ValueError(f"Step must be positive, got {step}")
    if start > end:
        return []

    if all(isinstance(value, int) for value in (start, end, step)):
        values: list[Number] = []
        current = int(start)
        while current <= int(end):
            values.append(current)
            current += int(step)
        if values and values[-1] != int(end) and values[-1] < int(end):
            values.append(int(end))
        return values

    values = []
    current = float(start)
    end_value = float(end)
    step_value = float(step)
    epsilon = 1e-12
    while current <= end_value + epsilon:
        values.append(round(current, 10))
        current += step_value
    if values and values[-1] < end_value - epsilon:
        values.append(round(end_value, 10))
    return values


def _to_vector(point: Sequence[Number] | Number) -> list[float]:
    if isinstance(point, (int, float)):
        return [float(point)]
    return [float(value) for value in point]


def _l1_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))
