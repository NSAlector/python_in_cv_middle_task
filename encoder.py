import numpy as np
import cv2
from typing import List, Optional, Tuple, Union
from torch.utils.data import Dataset
import torch
import os

class EncoderDataset(Dataset):
    """
    Кодировщик:
    grayscale -> resize 256x256 -> normalization [0,1] -> 16 патчей 64x64 -> автокорреляция (16x16) -> бинаризация T=0.5 -> шум p=20% -> int.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Union[torch.nn.Module, callable]] = None,
        noise_p: float = 0.20,
        patch_grid: Tuple[int, int] = (4, 4),
        patch_size: int = 64,
        threshold: float = 0.5
    ):
        self.image_paths = image_paths
        self.transform = transform
        self.noise_p = noise_p
        self.patch_rows, self.patch_cols = patch_grid
        self.num_patches = self.patch_rows * self.patch_cols
        self.patch_size = patch_size
        self.img_size = self.patch_size * self.patch_rows
        self.threshold = threshold
        self._precompute_encodings()
        self.encodings = [self._encode_single(path) for path in self.image_paths]

    def _precompute_encodings(self):
        self.encodings = []
        for path in self.image_paths:
            code = self._encode_single(path)
            self.encodings.append(code)

    def _encode_single(self, img_path_or_array: Union[str, np.ndarray]) -> bytes:
        if isinstance(img_path_or_array, str):
            img = cv2.imread(img_path_or_array, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"пропуск повреждённого: {os.path.basename(img_path_or_array)}")
                return b'\x00' * 32 #нулевой код
            img = cv2.resize(img, (self.img_size, self.img_size)).astype(np.float32) / 255.0
        
        #патчи 4x4=16
        patches = np.empty((self.num_patches, self.patch_size**2))
        for i in range(self.patch_rows):
            for j in range(self.patch_cols):
                idx = i * self.patch_cols + j
                row_start, col_start = i*self.patch_size, j*self.patch_size
                patches[idx] = img[row_start:row_start+self.patch_size, 
                                   col_start:col_start+self.patch_size].flatten()
        
        #автокорреляция
        autocorr = np.dot(patches, patches.T) / (self.patch_size**2)
        
        #бинаризация
        binary = (autocorr > self.threshold).astype(np.float32)
        
        #шум
        flat = binary.flatten()
        n = len(flat)
        flip_count = int(self.noise_p * n)
        if flip_count > 0:
            flips = np.random.choice(n, flip_count, replace=False)
            flat[flips] = 1 - flat[flips]
        

        bit_str = ''.join(str(int(b)) for b in flat)
        byte_array = np.packbits(np.array(list(bit_str), dtype=np.uint8))
        return byte_array.tobytes()  # bytes(32,)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        encoded_bytes = self.encodings[idx]
        img_tensor = torch.from_numpy(np.frombuffer(encoded_bytes, dtype=np.uint8).copy())
        
        if self.transform:
            img_tensor = self.transform(img_tensor)

        label = idx  
        return img_tensor, torch.tensor(label, dtype=torch.long)