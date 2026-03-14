import numpy as np
from PIL import Image


class ImageEncoder:
    def __init__(self, image_paths, noise_percent=20.0):
        """
        Args:
            image_paths: список путей к изображениям
            noise_percent: процент шума для бинаризованного вектора (p)
        """
        self.image_paths = image_paths
        self.noise_percent = noise_percent
        self.encoded_data = [self._encode_image(path) for path in image_paths]

    def __len__(self):
        """Возвращает количество изображений"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Возвращает закодированное изображение по индексу"""
        return self.encoded_data[idx]

    def _encode_image(self, image_path):
        """
        Кодирует одно изображение в целое число
        """
        try:
            # 1. Загрузка и преобразование
            img = Image.open(image_path).convert('L')

            # 2. Ресайз до 256x256
            img = img.resize((256, 256), Image.Resampling.LANCZOS)

            # 3. Нормализация [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0

            # 4. Разбиение на патчи 4x4 (16 патчей размером 64x64)
            patches = []
            patch_size = 64
            for i in range(0, 256, patch_size):
                for j in range(0, 256, patch_size):
                    patch = img_array[i:i + patch_size, j:j + patch_size]
                    patches.append(patch.flatten())

            patches = np.array(patches)

            # 5. Автокорреляция патчей (16x16 = 256 элементов)
            correlation_matrix = np.zeros((16, 16))
            for i in range(16):
                for j in range(16):
                    correlation_matrix[i, j] = np.corrcoef(patches[i], patches[j])[0, 1]

            correlation_vector = correlation_matrix.flatten()

            # 6. Бинаризация по порогу T=0.5
            binary_vector = (correlation_vector > 0.5).astype(np.int8)

            # 7. Зашумление
            if self.noise_percent > 0:
                n_noise = int(len(binary_vector) * self.noise_percent / 100)
                noise_indices = np.random.choice(len(binary_vector), n_noise, replace=False)
                binary_vector[noise_indices] = 1 - binary_vector[noise_indices]

            # 8. Преобразование в целое число
            key = sum(int(bit) * (2 ** i) for i, bit in enumerate(reversed(binary_vector)))

            return key

        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {e}")
            return 0

    def get_encoded_data(self):
        """Возвращает все закодированные данные"""
        return self.encoded_data

    def get_encoded_data_as_matrix(self):
        """
        Возвращает данные в виде матрицы для кластеризации
        """
        matrix = []
        for num in self.encoded_data:
            bits = [(num >> i) & 1 for i in range(256)]
            matrix.append(bits)
        return np.array(matrix, dtype=np.float32)


def get_encoder():
    return ImageEncoder
