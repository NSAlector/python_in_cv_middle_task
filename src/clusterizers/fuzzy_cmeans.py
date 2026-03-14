import numpy as np
from src.interfaces import Clusterizer

class FuzzyCMeans(Clusterizer):  
    def __init__(self, c, m=2.0, eps=1e-5, max_iter=100):
        self.c = c
        self.m = m
        self.eps = eps
        self.max_iter = max_iter
        self.centers = None
        self.u = None  

    def fit(self, X):  
        n = X.shape[0]
        self.u = np.random.rand(n, self.c)
        self.u = self.u / self.u.sum(axis=1, keepdims=True)
        
        for i in range(self.max_iter):
            u_old = self.u.copy()
            um = np.clip(self.u ** self.m, 1e-10, 1.0)
            um_sum = np.sum(um, axis=0, keepdims=True)
            self.centers = np.dot(um.T, X) / (um_sum + 1e-10)
            
            dist = np.zeros((n, self.c))
            for k in range(self.c):
                dist[:, k] = np.sum(np.abs(X - self.centers[k]), axis=1)
            dist = np.clip(dist, 1e-10, 1e10)
            
            power = np.clip(2 / (self.m - 1), 1.0, 10.0)
            u_new = 1.0 / (dist ** power)
            u_new_sum = np.sum(u_new, axis=1, keepdims=True)
            self.u = u_new / (u_new_sum + 1e-10)
            
            change = np.sum(np.abs(self.u - u_old))
            if change < self.eps:
                break
        
        labels = np.argmax(self.u, axis=1)
        return self.centers, labels
