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


manualSeed = -1

if manualSeed>0 :
    print("Manual Seed : ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)

#데이터 파일 경로
dataroot = ""

#워커
workers = 0

#GPU개수
ngpu = 1

#배치 사이즈
batch_size = 8

#이미지 크기
image_size = 256

#이미지 채널 갯수(RGB = 3)
nc = 3

#ganarator 입력값 갯수
#월, 일, 시, 분, 온도, 습도, 광도(7개), Recent Image(16*16=256), Random
nz = 264

#출력값
ngf = 256
ndf = 256
device1 = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
#device2 = torch.device("cpu")
device2 = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


#Generator
class Generator(nn.Module):
    def __init__(self,ngpu):
        super(Generator,self).__init__()
        self.ngpu = ngpu
        #신경망 구성
        self.main = nn.Sequential(
            #ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
            #in_channels ( int ) – 입력 이미지의 채널 수 = nz 
            #out_channels ( int ) – 컨볼루션에 의해 생성된 채널 수 = ngf*8
            #kernel_size( int 또는 tuple ) – 컨볼루션 커널의 크기 = 4
            #stride ( int 또는 tuple , optional) – 컨볼루션의 보폭 = 1
            #padding = 0

            #잘 섞어주는 레이어
            nn.Linear(nz, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            nn.Linear(128, nz),
            nn.BatchNorm1d(nz),
            nn.ReLU(inplace=False))
        self.conv=nn.Sequential(
            # 입력(nz)-> 4x4x(ngf*8)
            nn.ConvTranspose2d(nz, ngf * 8,      16, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 8, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf,     4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc,          4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        out = self.main(input)
        out = out.reshape(batch_size, -1, 1, 1)
        out = self.conv(out)
        return out
    
h = None
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nn_layers = nn.ModuleList()

        #이미지 먼저 처리
        self.nn_layers.append(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        self.nn_layers.append(nn.BatchNorm2d(ndf))
        self.nn_layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.nn_layers.append(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))
        self.nn_layers.append(nn.BatchNorm2d(ndf * 2))
        self.nn_layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.nn_layers.append(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        self.nn_layers.append(nn.BatchNorm2d(ndf * 4))
        self.nn_layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.nn_layers.append(nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False))
        self.nn_layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        #10
        self.nn_layers.append(nn.Linear(265, 256))
        self.nn_layers.append(nn.ReLU(inplace=False))
        self.nn_layers.append(nn.Linear(256, 128))
        self.nn_layers.append(nn.ReLU(inplace=False))
        self.nn_layers.append(nn.Linear(128, 64))
        self.nn_layers.append(nn.ReLU(inplace=False))
        self.nn_layers.append(nn.Linear(64, 32))
        self.nn_layers.append(nn.ReLU(inplace=False))
        self.nn_layers.append(nn.Linear(32, 1))
        self.nn_layers.append(nn.Sigmoid())
        #self.conv[4] = nn.Sigmoid()

    def forward(self, input, additional_data):
        global h
        h=input
        for layer_number in range(11):
            h=self.nn_layers[layer_number](h)
        #print(h.shape)
        h=h.reshape(batch_size,-1)
        for data in additional_data:
            d=data.reshape(-1,1).to(device2)
            h = torch.cat([h,d], dim=1)
        
        for layer_number in range(11,len(self.nn_layers)):
            h=self.nn_layers[layer_number](h)
        
        return h



