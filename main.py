import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from plugin_loader import PluginLoader


def load_dataset(dataset_path):
    """
    Загружает список изображений из директории
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(dataset_path, ext)))
        image_paths.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))

    print(f"Найдено {len(image_paths)} изображений в {dataset_path}")
    return image_paths


def visualize_clusters(x_encoded, labels, centers, title="Результаты кластеризации"):
    """
    Визуализация результатов кластеризации
    """
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_encoded)

    plt.figure(figsize=(15, 5))

    # График 1: PCA проекция
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=labels, cmap='viridis',
                          alpha=0.6, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter)
    plt.title(f'{title} (n_clusters={len(np.unique(labels))})')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    # График 2: Центры кластеров
    plt.subplot(1, 2, 2)
    centers_pca = pca.transform(centers)
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=labels, cmap='viridis',
                alpha=0.4, edgecolors='black', linewidth=0.5)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X',
                s=200, edgecolors='black', linewidth=2, label='Центры')
    plt.title('Центры кластеров')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()

    plt.tight_layout()
    plt.savefig('clustering_results.png', dpi=150)
    plt.show()

def main():
    """Основная функция тестирования"""

    dataset_path = "./dataset"

    print("=" * 60)
    print("ЗАПУСК ТЕСТОВОГО СКРИПТА")
    print("=" * 60)

    # 1. Загрузка плагинов
    print("\n1. Загрузка плагинов...")
    loader = PluginLoader()

    encoder_class = loader.load_encoder('encoder')
    clusterizer_class = loader.load_clusterizer('clusterizer')
    optimizer_class = loader.load_optimizer('optimizer')

    if encoder_class is None:
        print("Ошибка: Не удалось загрузить кодировщик!")
        return

    if clusterizer_class is None:
        print("Ошибка: Не удалось загрузить кластеризатор!")
        return

    if optimizer_class is None:
        print("Ошибка: Не удалось загрузить оптимизатор!")

    # 2. Загрузка и кодирование датасета
    print("\n2. Загрузка и кодирование датасета...")
    image_paths = load_dataset(dataset_path)

    if not image_paths:
        print(f"Ошибка: Датасет пуст: {dataset_path}")
        return

    encoder = encoder_class(image_paths, noise_percent=20.0)

    encoded_data = encoder.get_encoded_data()
    x_matrix = encoder.get_encoded_data_as_matrix()

    print(f"Закодировано {len(encoded_data)} изображений")
    print(f"Размерность признаков: {x_matrix.shape[1]}")
    print(f"Пример закодированных значений: {encoded_data[:3]}")

    print(f"\nСтатистика данных:")
    print(f"  Среднее: {np.mean(x_matrix):.4f}")
    print(f"  Стандартное отклонение: {np.std(x_matrix):.4f}")
    print(f"  Минимум: {np.min(x_matrix)}")
    print(f"  Максимум: {np.max(x_matrix)}")

    # 3. Оптимизация
    print("\n3. Оптимизация")

    param_grid = {
        'n_clusters': [2, 3, 4, 5],
        'm': [1.5, 2.0, 2.5],
        'eps': [0.0001, 0.001],
        'max_iter': [100],
    }

    delta = {
        'n_clusters': 1,
        'm': 0.5,
        'eps': 0.0005,
    }

    print("\n" + "=" * 50)
    print(f"param_grid: {param_grid}")
    print(f"delta: {delta}")
    print("=" * 50)

    try:
        optimizer = optimizer_class(param_grid, delta)
        best_params, best_clusterizer = optimizer.optimize(clusterizer_class, x_matrix)
        print(f"Оптимизация успешна! Лучшие параметры: {best_params}")
        print(f"Лучшая оценка: {optimizer.best_score:.4f}")
    except Exception as e:
        print(f"Оптимизация не удалась: {e}")
        return

    # 4. Анализ результатов
    print("\n4. Анализ результатов")

    labels = best_clusterizer.get_labels()
    centers = best_clusterizer.get_centers()
    membership = best_clusterizer.get_membership()

    unique, counts = np.unique(labels, return_counts=True)

    if len(unique) > 1:
        try:
            sil_score = silhouette_score(x_matrix, labels, metric='manhattan')
            print(f"Коэффициент силуэта: {sil_score:.4f}")
        except Exception as e:
            print(f"Не удалось вычислить коэффициент силуэта: {e}")

    max_membership = np.max(membership, axis=1)
    print(f"Средняя максимальная принадлежность: {np.mean(max_membership):.4f}")
    print(f"Медианная принадлежность: {np.median(max_membership):.4f}")
    print(f"Доля точек с нечеткой принадлежностью (<0.7): {np.mean(max_membership < 0.7) * 100:.1f}%")

    # 5. Визуализация
    print("\n5. Визуализация")
    try:
        visualize_clusters(x_matrix, labels, centers,
                           f"Fuzzy C-Means (m={best_params.get('m', 2.0)})")
    except Exception as e:
        print(f"Ошибка при визуализации: {e}")

    # 6. Сохранение результатов
    print("\n6. Сохранение результатов...")

    try:
        with open('clustering_results.txt', 'w', encoding='utf-8') as f:
            f.write("РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Количество изображений: {len(encoded_data)}\n")
            f.write(f"Использованные параметры: {best_params}\n\n")

            f.write("Распределение по кластерам:\n")
            for cluster_id, count in zip(unique, counts):
                f.write(f"  Кластер {cluster_id}: {count} элементов\n")

            f.write(f"\nСредняя принадлежность: {np.mean(max_membership):.4f}\n")

            f.write("\nЗакодированные значения (первые 20):\n")
            for i, val in enumerate(encoded_data[:20]):
                f.write(f"  Изображение {i + 1}: {val}\n")

        print("Результаты сохранены в clustering_results.txt")
        print("Визуализация сохранена в clustering_results.png")

    except Exception as e:
        print(f"Ошибка при сохранении результатов: {e}")

    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)


if __name__ == "__main__":
    main()
