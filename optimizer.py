from base_optimizer import AbstractOptimizerMeta
from typing import Dict, Tuple, Any
import numpy as np
from itertools import product

class GridSearchOptimizer(AbstractOptimizerMeta, name="grid"):
    """
    Оптимизатор гиперпараметров
    """
    
    def compute_within_cluster_variance(self, data: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
        total_variance = 0
        for k in range(len(centers)):
            cluster_points = data[labels == k]
            if len(cluster_points) > 0:
                distances = np.sum(np.abs(cluster_points - centers[k]), axis=1)
                total_variance += np.sum(distances)
        return total_variance
    
    def generate_grid(self, param_ranges: Dict[str, Tuple[float, float, float]]) -> list:
        grids = []
        param_names = []
        
        for param, (min_val, max_val, delta) in param_ranges.items():
            values = np.arange(min_val, max_val + delta/2, delta)
            grids.append(values)
            param_names.append(param)
        
        #все комбинации
        grid_combinations = list(product(*grids))
        return [dict(zip(param_names, combo)) for combo in grid_combinations]
    
    def optimize(
        self,
        clusterer_factory,
        data: np.ndarray,
        param_ranges: Dict[str, Tuple[float, float, float]],
    ) -> Tuple[Dict[str, Any], Tuple[np.ndarray, np.ndarray]]:
        grid = self.generate_grid(param_ranges)
        best_score = float('inf')
        best_params = None
        best_result = None
        
        print(f"Перебор {len(grid)} комбинаций...")
        
        for i, params in enumerate(grid):
            clusterer = clusterer_factory()
        for param, value in params.items():
            if param == 'c':
                setattr(clusterer, param, int(value))
            else:
                setattr(clusterer, param, value)
            
            #кластеризуем
            centers, labels = clusterer.fit(data)
            
            #критерий: минимум разброса при минимуме кластеров
            num_clusters = len(centers)
            variance = self.compute_within_cluster_variance(data, centers, labels)
            score = variance * num_clusters
            
            print(f"Комбинация {i+1}: {params}, кластеров: {num_clusters}, разброс: {variance:.1f}")
            
            if score < best_score:
                best_score = score
                best_params = params
                best_result = (centers, labels)
        
        print(f"Лучшие параметры: {best_params}, score: {best_score}")
        return best_params, best_result
