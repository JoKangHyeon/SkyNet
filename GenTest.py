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

from AI import Generator

resume_G = r"D:\Project\model\ver2.1.G.pt"
prefix = "ver2"
nz = 264
netG = torch.load(resume_G)
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def show_img(cnt):
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    for i in range(64):
        sub = fig.add_subplot(8,8,i+1)
        plt.axis("off")
        plt.imshow(np.transpose(img_list[i],(1,2,0)))
        
    #ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    #ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    #plt.show()
    plt.savefig('C:\\Users\\jokh9\\Desktop\\t\\%d.png'%(cnt))

img_list = []


for k in range(8):
    img_list=[]
    for i in range(8):
        with torch.no_grad():
            netG.eval()
            noises = torch.randn(nz-9)
            noises = torch.cat([noises, torch.FloatTensor([2023,9,18,14,6,21,0,(300/7*i),(300/7*k)])]).to(device).reshape(1,-1)
            for j in range(7):
                noise = torch.randn(nz-9)
                noise = torch.cat([noise, torch.FloatTensor([2023,9,18,14,6,21,(255/7*j),(300/7*i),(300/7*k)])]).to(device).reshape(1,-1)
                noises = torch.cat([noises,noise])
            batch = netG(noises).detach().cpu()
            for j in range(8):
                img_list.append((batch[j]+1)/2)
    show_img(k)

#show_img()
