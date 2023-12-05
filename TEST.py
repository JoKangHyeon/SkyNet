import os
import torch
import torch.nn as nn
import numpy as np

from glob import glob

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.optim as optim

import matplotlib.pyplot as plt
import PIL
import matplotlib.animation as animation

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms=None):
            self.root_dir = root_dir
            self.classes = os.listdir(self.root_dir)
            self.transforms = transforms
            self.data = []
            self.labels = []
            
            for idx, cls in enumerate(self.classes):
                cls_dir = os.path.join(self.root_dir, cls)
                for img in glob(os.path.join(cls_dir, '*.png')):
                    self.data.append(img)
                    self.labels.append(idx)

    def __getitem__(self, idx):
        img_path, label = self.data[idx], self.labels[idx]
        img = PIL.Image.open(img_path)
        
        if self.transforms:
        	img = self.transforms(img)
        filename = os.path.basename(img_path)
        sample = {'image':img, 'label':label, 'addition_data' : list(map(int,filename.split('.')[0].split("-")))}
        
        return sample
    
    def __len__(self):
    	return len(self.data)

dataroot = r"D:\Project\crop"
#이미지 크기
image_size = 256

#배치 사이즈
batch_size = 64

#워커
workers = 0

dataset = CustomDataset(root_dir=dataroot,
                           transforms=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

dataset=[]
for data in dataloader:
    for i in range(len(data['image'])):
        dataset.append(torch.mean(data['image'][i][0][0]))
            
