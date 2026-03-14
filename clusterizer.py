from base_clusterizer import AbstractClustererMeta
import numpy as np
from typing import Tuple

class FuzzyCMeans(AbstractClustererMeta, name="fuzzy"):
    """
    Кластеризатор:
    алгоритм Нечеткие C-средние
    """
    
    def __init__(self, c: int = 3, m: float = 2.0, eps: float = 1e-5, max_iter: int = 100):
        super().__init__()
        self.c = int(c)   
        self.m = float(m)
        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.centers_ = None
        self.labels_ = None
    
    def _l1_distance(self, point: np.ndarray, center: np.ndarray) -> float:
        return np.sum(np.abs(point - center))
    
    def _update_membership(self, data: np.ndarray, centers: np.ndarray) -> np.ndarray:
        n_samples = data.shape[0]
        u = np.zeros((n_samples, self.c))
        
        for i in range(n_samples):
            distances = np.array([self._l1_distance(data[i], centers[j]) 
                                for j in range(self.c)])
            distances = np.where(distances == 0, 1e-10, distances)
            distances = distances ** (-2 / (self.m - 1))
            u[i] = distances / np.sum(distances)
        
        return u
    
    def _update_centers(self, data: np.ndarray, u: np.ndarray) -> np.ndarray:
        um = u ** self.m
        n_samples, n_features = data.shape
        
        centers = np.zeros((int(self.c), n_features))
        for j in range(int(self.c)):
            weight_total = np.sum(um[:, j])
            if weight_total > 0:
                centers[j] = np.sum(um[:, j][:, np.newaxis] * data, axis=0) / weight_total
            else:
                centers[j] = data[0]
        
        return centers
    
    def fit(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples, n_features = data.shape
        
        #инициализация случайными коэффициентами принадлежности
        u = np.random.uniform(0.001, 0.999, (n_samples, int(self.c)))
        u /= np.sum(u, axis=1, keepdims=True)
        
        for iteration in range(self.max_iter):
            u_old = u.copy()
            
            centers = self._update_centers(data, u)
            
            u = self._update_membership(data, centers)
            
            change = np.sum(np.abs(u - u_old))
            if change < self.eps:
                print(f"сошлось на итерации: {iteration+1}")
                break

        self.centers_ = centers
        self.labels_ = np.argmax(u, axis=1)
        
        return centers, self.labels_