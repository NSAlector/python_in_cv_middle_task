import cv2
import numpy as np
import random
import glob


class Encoder:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = []
        self.load_data()

    def load_data(self):
        image_files = glob.glob(self.data_dir + '/*.png')
        if not image_files:
            raise ValueError(f"No images in {self.data_dir}")
        for img_path in image_files:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            self.data.append(img)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of range")
        img = self.data[idx]
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0
        patches = []

        for i in range(4):
            for j in range(4):
                patch = img[i*64:(i+1)*64, j*64:(j+1)*64]
                flat = patch.flatten()
                patches.append(flat)

        patches = np.array(patches)  
        corr_vec = []
        for i in range(16):
            for j in range(16):
                dot = np.sum(patches[i] * patches[j])
                norm_i = np.linalg.norm(patches[i])
                norm_j = np.linalg.norm(patches[j])
                corr = dot / (norm_i * norm_j + 1e-8)
                corr_vec.append(corr)

        corr_vec = np.array(corr_vec)
        binary = (corr_vec > 0.5).astype(int)
        n_noise = int(0.2 * len(binary))
        noise_idxs = random.sample(range(len(binary)), n_noise)
        binary[noise_idxs] = 1 - binary[noise_idxs]
        binary_str = ''.join(binary.astype(str))
        key = int(binary_str, 2)
        
        return key