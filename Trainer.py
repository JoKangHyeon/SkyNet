import AI
from AI import Generator
from AI import Discriminator

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
from PIL import Image


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

#learning rate
lr = 0.0002

# Beta1 Adam에 쓰는 거라고 함
beta1 = 0.5

#GPU개수
ngpu = 1


#배치 사이즈
batch_size = 8
# 에포크
num_epochs = 16

#이미지 크기
image_size = 256
#ganarator 입력값 갯수
#월, 일, 시, 분, 온도, 습도, 광도(7개), Recent Image(16*16=256), Random
nz = 264

#출력값
ngf = 256
ndf = 256

device1 = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
#device2 = torch.device("cpu")
device2 = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

dataroot = r"D:\Project\crop"
#root_dir, transforms=None):



# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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

dataset = CustomDataset(root_dir=dataroot,
                           transforms=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch['image'][0].to(device2)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.waitforbuttonpress(5)
# Create the generator
netG = AI.Generator(ngpu).to(device1)

# Handle multi-GPU if desired
if (device1.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
#  to ``mean=0``, ``stdev=0.02``.
netG.apply(weights_init)

# Print the model
print(netG)

# Create the Discriminator
netD = AI.Discriminator(ngpu).to(device2)


if (device1.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
# like this: ``to mean=0, stdev=0.2``.
netD.apply(weights_init)

# Print the model
print(netD)

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = None
inited = False

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))



# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0


def show_img():
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    plt.show()

def plt_show():
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def print_all():
    for i in range(len(img_list)):
        im = np.transpose(img_list.pop().numpy(),(1,2,0))
        im = im*255
        im = im.astype(np.uint8)
        im = Image.fromarray(im)
        im.save('D:\\Project\\output\\%d.png'%(i))


resume = True
resume_D = r"D:\Project\model\ver2.16.D.pt"
resume_G = r"D:\Project\model\ver2.16.G.pt"
prefix = "ver3"

if resume:
    netD = torch.load(resume_D)
    netG = torch.load(resume_G)

    


print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs+1):
    i=0
    # For each batch in the dataloader
    for data in dataloader:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            current_data = data['addition_data']
            real_cpu = data['image'].to(device2)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device2)
            # Forward pass real batch through D
            output = netD(real_cpu,current_data).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, nz-9,device=device1)
            for cd in current_data:
                d=cd.reshape(-1,1).to(device2)
                noise = torch.cat([noise,d], dim=1)

            if not inited:
                fixed_noise = torch.tensor(noise)
                inited=True
                
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach().to(device2),current_data).view(-1).to(device2)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake.to(device2), current_data).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if iters % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(data['image']),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 2 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
            i+=1
    torch.save(netD,r'D:\Project\model\{0}.{1}.D.pt'.format(prefix,epoch))
    torch.save(netG,r'D:\Project\model\{0}.{1}.G.pt'.format(prefix,epoch))
    print("Epoch {0} end, save".format(epoch))


