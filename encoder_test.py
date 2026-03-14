import torch
from encoder import EncoderDataset

paths = ['image1.jpg']
ds = EncoderDataset(paths)

item = ds[0]
print(type(item[0]), item[0].shape, item[0].dtype)
print(item[0].numpy().tobytes().hex())
