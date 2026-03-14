import glob, importlib.util, os
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.interfaces import Clusterizer, HyperparamOptimizerInterface
from src.encoders import Encoder

def load_clusterizers():
    clusterizers = []
    for file in glob.glob("src/clusterizers/*.py"):
        if file.endswith("__init__.py"):
            continue
        module_name = "src.clusterizers." + os.path.basename(file)[:-3]
        spec = importlib.util.spec_from_file_location(module_name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for attr in module.__dict__.values():
            if isinstance(attr, type) and issubclass(attr, Clusterizer) and attr is not Clusterizer:
                clusterizers.append(attr)
    return clusterizers

def load_optimizers():
    optimizers = []
    for file in glob.glob("src/optimizers/*.py"):
        if file.endswith("__init__.py"):
            continue
        module_name = "src.optimizers." + os.path.basename(file)[:-3]
        spec = importlib.util.spec_from_file_location(module_name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for attr in module.__dict__.values():
            if isinstance(attr, type) and issubclass(attr, HyperparamOptimizerInterface) and attr is not HyperparamOptimizerInterface:
                optimizers.append(attr)
    return optimizers

def main():
    encoder = Encoder("./dataset")
    encoded = [encoder[i] for i in range(len(encoder))]
    print(f"Закодировано {len(encoded)} изображений")
    print("Пример кодировки:", encoded[0])

    clusterizers = load_clusterizers()
    print("Кластеризаторы:", [c.__name__ for c in clusterizers])
    optimizers = load_optimizers()
    print("Оптимизаторы:", [o.__name__ for o in optimizers])

    ClusterizerClass = clusterizers[0]      
    OptimizerClass  = optimizers[0]         

    optimizer = OptimizerClass(ClusterizerClass, encoded)
    param_ranges = {
        'c': (2, 5, 1),
        'm': (1.5, 2.5, 0.1),
        'eps': (1e-6, 1e-3, 1e-5),
    }
    best_params, centers, labels = optimizer.grid_search(param_ranges)

    print("Оптимальные параметры:", best_params)
    print("Центры кластеров:", centers)
    print("Распределение:")
    for k in range(len(centers)):
        print(f"Кластер {k}: {np.sum(labels == k)} изображений")

if __name__ == "__main__":
    main()