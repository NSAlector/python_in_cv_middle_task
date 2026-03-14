import numpy as np
from typing import Tuple, Dict, Any
from src.interfaces import HyperparamOptimizerInterface

def objective(centers, X, labels):
    total_score = 0
    for k in range(len(centers)):
        mask = labels == k
        if np.sum(mask) == 0:
            continue
        cluster_points = X[mask]
        center_k = centers[k]
        score = np.sum(np.abs(cluster_points - center_k))
        total_score += score
    return total_score

class HyperparamOptimizer(HyperparamOptimizerInterface):  
    def __init__(self, clusterer_class, data):
        self.clusterer_class = clusterer_class
        data_array = np.array(data, dtype=float)
        self.X = (data_array - data_array.min()) / (data_array.max() - data_array.min() + 1e-8)
        self.X = self.X.reshape(-1, 1)

    def grid_search(self, param_ranges: Dict[str, Tuple[Any, Any, Any]]) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]: 
        best_score = float('inf')
        best_params = None
        best_centers = None
        best_labels = None
        
        c_range = param_ranges['c']
        m_range = param_ranges['m'] 
        eps_range = param_ranges['eps']
        
        c_min, c_max, c_delta = c_range
        m_min, m_max, m_delta = m_range
        eps_min, eps_max, eps_delta = eps_range
        
        for c in range(c_min, c_max + 1, c_delta):
            for m in np.arange(m_min, m_max, m_delta):
                for eps in np.arange(eps_min, eps_max, eps_delta):
                    fcm = self.clusterer_class(c=c, m=m, eps=eps)
                    centers, labels = fcm.fit(self.X)
                    score = objective(centers, self.X, labels)
                    if score < best_score:
                        best_score = score
                        best_params = {'c': c, 'm': m, 'eps': eps}
                        best_centers = centers
                        best_labels = labels
        return best_params, best_centers, best_labels
