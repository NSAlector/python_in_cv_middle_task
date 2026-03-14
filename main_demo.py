"""
Демонстрационный скрипт, показывающий работу всех модулей:
- загрузка изображений через ImageEncoder
- динамическая загрузка плагинов кластеризаторов и оптимизаторов
- оптимизация гиперпараметров для нечётких C-средних
- вывод результатов и визуализация
"""

import os
import numpy as np
from encoder import ImageEncoder
from clusterizer import load_plugins as load_clusterizers, FuzzyCMeansClusterizer
from optimizer import load_plugins as load_optimizers, GridSearchOptimizer

def main():
    # 1. Загружаем данные
    dataset_path = "./dataset"
    if not os.path.isdir(dataset_path):
        print(f"Ошибка: папка {dataset_path} не найдена.")
        return

    print("Загрузка изображений...")
    encoder = ImageEncoder(dataset_path, seed=42)
    print(f"Найдено изображений: {len(encoder)}")

    if len(encoder) == 0:
        print("Нет изображений для обработки.")
        return

    # Преобразуем все изображения в матрицу данных
    X = np.array([encoder[i] for i in range(len(encoder))])
    print(f"Размер данных: {X.shape}")  # (N, 256)

    # 2. Динамическая загрузка плагинов кластеризаторов
    print("\n--- Загрузка плагинов кластеризаторов ---")
    clusterizer_classes = load_clusterizers()
    print(f"Найдено классов кластеризаторов: {len(clusterizer_classes)}")
    for cls in clusterizer_classes:
        print(f"  - {cls.__name__}")

    # Выбираем конкретный класс для оптимизации
    target_clusterizer = FuzzyCMeansClusterizer

    # 3. Динамическая загрузка плагинов оптимизаторов
    print("\n--- Загрузка плагинов оптимизаторов ---")
    optimizer_classes = load_optimizers()
    print(f"Найдено классов оптимизаторов: {len(optimizer_classes)}")
    for cls in optimizer_classes:
        print(f"  - {cls.__name__}")

    # Выбираем GridSearchOptimizer
    optimizer = GridSearchOptimizer(verbose=True)

    # 4. Задаём сетку параметров
    param_grid = {
        'c': (2, 5, 1),          # от 2 до 5 кластеров с шагом 1
        'm': (1.5, 3.0, 0.5)      # степень размытости
    }

    print("\n--- Запуск оптимизации ---")
    best_params, best_result, best_score = optimizer.optimize(
        X, target_clusterizer, param_grid
    )

    print("\n=== РЕЗУЛЬТАТ ОПТИМИЗАЦИИ ===")
    print(f"Лучшие параметры: {best_params}")
    print(f"Лучший score (сумма взвешенных расстояний): {best_score:.4f}")

    centers, memberships = best_result
    print(f"Форма центров: {centers.shape}")
    print(f"Форма матрицы принадлежностей: {memberships.shape}")

    # 5. Анализ результатов
    hard_labels = np.argmax(memberships, axis=1)
    unique, counts = np.unique(hard_labels, return_counts=True)
    print("Распределение точек по кластерам:")
    for cl, cnt in zip(unique, counts):
        print(f"  Кластер {cl}: {cnt} точек")

    # 6. Визуализация с понижением размерности (PCA)
    try:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        plt.figure(figsize=(8,6))
        scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=hard_labels, cmap='viridis', alpha=0.7)
        plt.scatter(centers[:,0], centers[:,1], c='red', marker='X', s=200, label='Centers')
        plt.colorbar(scatter, label='Cluster')
        plt.title(f"Кластеризация (PCA) с параметрами {best_params}")
        plt.legend()
        plt.savefig("clustering_result.png")
        plt.show()
        print("График сохранён в clustering_result.png")
    except ImportError:
        print("Для визуализации установите scikit-learn и matplotlib")

if __name__ == "__main__":
    main()