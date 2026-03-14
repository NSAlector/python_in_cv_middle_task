import numpy as np
import cv2
import os
from encoder import EncoderDataset
import clusterizer  # регистрация кластеризатора
import optimizer   # регистрация оптимизатора
from clusterizer import AbstractClustererMeta
from base_optimizer import AbstractOptimizerMeta
np.random.seed(0)

def create_test_data(n=30):
    paths = []
    for i in range(n):
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        path = f'test_{i}.png'
        cv2.imwrite(path, img)
        paths.append(path)
    return paths


def load_dataset_from_folder(folder_path: str) -> list:
    paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(root, file))
    paths.sort()
    print(f"Загружено {len(paths)} изображений из {folder_path}")
    return paths


if __name__ == "__main__":
    #загружаем/делаем данные
    # paths = create_test_data()
    folder = r'D:\AZAT_files\MIPT\SUBJECTS\Semestr 9\PythonLections\02_lection\blood_cells_dataset\BCCD Dataset with mask\train\original'
    paths = load_dataset_from_folder(folder)

    ds = EncoderDataset(paths)
    codes = np.stack([ds[i][0].numpy() for i in range(len(ds))])
    print("Уникальных кодов:", len(np.unique(codes, axis=0)))
    print(f"Данные: {codes.shape}")
   
    #диапазоны гиперпараметров
    param_ranges = {
        'c': (2, 6, 1),    # 2,3,4,5,6 кластеров
        'm': (1.5, 2.5, 0.5),  # 1.5, 2.0, 2.5
        'eps': (1e-4, 1e-3, 5e-4)
    }
    
    #Оптимизатор
    optimizer = AbstractOptimizerMeta.get_plugin("grid")
    best_params, (best_centers, best_labels) = optimizer.optimize(AbstractClustererMeta.get_plugin("fuzzy"), codes, param_ranges)
    

    unique_labels, counts = np.unique(best_labels, return_counts=True)
    print(f"Кластеры: {dict(zip(unique_labels, counts))}")
    
    #очистка если создавали данные, а не загружали
    # for p in paths: os.remove(p)
