import numpy as np
from encoder import EncoderDataset
import clusterizer 
from base_clusterizer import AbstractClustererMeta
import os
import cv2

#тестовые изображения
def create_test_images(n=10):
    img_base = np.zeros((256, 256), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                img_base[i*64:(i+1)*64, j*64:(j+1)*64] = 255
    paths = []
    for k in range(n):
        path = f'test_img_{k}.png'
        cv2.imwrite(path, img_base + np.random.randint(0, 50, (256, 256)))
        paths.append(path)
    return paths


if __name__ == "__main__":
    paths = create_test_images(20)
    ds = EncoderDataset(paths)
    codes = np.stack([ds[i][0].numpy() for i in range(len(ds))])
    print(f"Данные: {codes.shape}")
    
    clusterer = AbstractClustererMeta.get_plugin("fuzzy")
    clusterer.c = 4  #колво кластеров
    
    centers, labels = clusterer.fit(codes)
    
    print(f"Кластеров: {len(centers)}")
    print(f"Метки: {np.unique(labels, return_counts=True)}")
    
    #удаление тестовых изображений
    for p in paths: os.remove(p)
