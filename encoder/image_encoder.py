import os
import cv2
import numpy as np

class ImageEncoder:
    """
    Кодировщик изображений: преобразует каждое изображение в бинарный вектор длины 256.
    Поддерживает доступ по индексу (__getitem__) и возвращает numpy.ndarray формы (256,) с элементами 0/1.
    """

    def __init__(self, dataset_path, p_noise=0.2, seed=None):
        """
        dataset_path : str   – путь к папке с PNG-изображениями
        p_noise      : float – доля инвертируемых битов (по умолчанию 0.2)
        seed         : int   – базовое значение для генератора шума (обеспечивает воспроизводимость)
        """
        self.dataset_path = dataset_path
        self.p_noise = p_noise
        self.base_seed = seed

        # Получаем список всех PNG-файлов в папке (регистронезависимо)
        self.image_files = [f for f in os.listdir(dataset_path)
                            if f.lower().endswith('.png')]
        self.image_files.sort()  # фиксируем порядок

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Возвращает бинарный вектор (0/1) для idx-го изображения.
        """
        # Полный путь к файлу
        img_path = os.path.join(self.dataset_path, self.image_files[idx])

        # Шаг 1–3: загрузка, grayscale, ресайз 256x256, нормализация в [0,1]
        img = self._load_and_preprocess(img_path)

        # Шаг 4: декомпозиция на 16 патчей 64x64 и их развёртка
        patches = self._extract_patches(img)          # список из 16 массивов длины 4096

        # Шаг 5: автокорреляция патчей → вектор 256
        corr_vector = self._compute_correlation(patches)

        # Шаг 6: бинаризация по порогу 0.5
        binary = (corr_vector >= 0.5).astype(np.uint8)

        # Шаг 7: зашумление (инвертирование p% битов)
        noisy = self._add_noise(binary, idx)

        return noisy

    def get_key(self, idx):
        """
        Преобразует бинарный вектор изображения в целое число (шаг 8).
        Интерпретация: первый элемент вектора – старший бит (big-endian).
        """
        bits = self[idx]                       # бинарный вектор (numpy array)
        # Собираем целое число сдвигами
        key = 0
        for b in bits:
            key = (key << 1) | int(b)
        return key

    # ---------- Вспомогательные методы (инкапсуляция шагов) ----------

    def _load_and_preprocess(self, path):
        """
        Загружает изображение, преобразует в оттенки серого,
        изменяет размер до 256x256 и нормализует пиксели в [0,1].
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {path}")

        # Ресайз до 256x256
        img = cv2.resize(img, (256, 256))

        # Нормализация: [0,255] -> [0,1]
        img = img.astype(np.float32) / 255.0
        return img

    def _extract_patches(self, img):
        """
        Разбивает изображение 256x256 на сетку 4x4 патчей размером 64x64.
        Каждый патч разворачивается в одномерный массив (длина 4096).
        Возвращает список из 16 таких массивов.
        """
        patches = []
        patch_size = 64
        for i in range(0, 256, patch_size):
            for j in range(0, 256, patch_size):
                patch = img[i:i+patch_size, j:j+patch_size]
                patches.append(patch.flatten())
        # Проверка: должно быть ровно 16 патчей
        assert len(patches) == 16, f"Ожидалось 16 патчей, получено {len(patches)}"
        return patches

    def _compute_correlation(self, patches):
        """
        Вычисляет попарные корреляции Пирсона между 16 патчами.
        Возвращает развёрнутую корреляционную матрицу размером 256.
        """
        # Строим матрицу "патчи × пиксели" (16 строк, 4096 столбцов)
        X = np.array(patches)               # shape (16, 4096)

        # Матрица корреляций 16x16
        corr_matrix = np.corrcoef(X)        # shape (16, 16)

        # Замена NaN (возникают при нулевой дисперсии патча) на 0
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Разворачиваем в вектор 256 (порядок row-major)
        corr_vector = corr_matrix.flatten()
        return corr_vector

    def _add_noise(self, binary_vector, idx):
        """
        Инвертирует случайные p% битов в бинарном векторе.
        Если указан seed, шум генерируется детерминированно на основе (base_seed + idx).
        """
        noisy = binary_vector.copy()
        n_noise = int(len(noisy) * self.p_noise)   # количество бит для инвертирования

        # Создаём генератор случайных чисел (свой для каждого изображения, если задан seed)
        if self.base_seed is not None:
            rng = np.random.default_rng(self.base_seed + idx)
        else:
            rng = np.random.default_rng()

        # Выбираем уникальные индексы
        indices = rng.choice(len(noisy), size=n_noise, replace=False)
        noisy[indices] = 1 - noisy[indices]
        return noisy

# ---- Простой тест при запуске файла напрямую ----
if __name__ == "__main__":
    # Укажите путь к папке с изображениями (пример)
    test_path = "./dataset"
    if os.path.exists(test_path):
        encoder = ImageEncoder(test_path, seed=42)
        print(f"Найдено изображений: {len(encoder)}")
        if len(encoder) > 0:
            vec = encoder[0]
            print(f"Вектор для первого изображения: {vec[:10]}... (длина {len(vec)})")
            key = encoder.get_key(0)
            print(f"Целое число (ключ): {key}")
    else:
        print(f"Папка {test_path} не найдена. Замените путь на корректный для теста.")