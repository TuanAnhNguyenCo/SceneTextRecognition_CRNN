import torchvision
from torch.nn import functional as F
from torchvision import models , transforms
from torch.utils.data import Dataset , DataLoader
import torch
import torch.nn as nn
from PIL import Image 




class STRDataset(Dataset): 
    def __init__(self ,X, y,char_to_idx , max_label_len , label_encoder=None, transform=None):
        self.transform = transform
        self.img_paths = X
        self.labels = y
        self.char_to_idx = char_to_idx 
        self.max_label_len = max_label_len
        self.label_encoder = label_encoder
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.label_encoder:
            encoded_label , label_len = self.label_encoder(label , self.char_to_idx , self.max_label_len)
        return img , encoded_label , label_len





