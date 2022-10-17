
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-11


< cGAN(Conditional Generative Adversarial Networks) ���� >

 - Conditional Generative Adversarial Nets (Arxiv 2014)


< Condition Vector>

 - Mnist data�� class (0~9)�� ���� one hot vector�� condition vector�� Ȱ�� 

 - ������ GAN �� ������ ��� ���������� �Ʒ��� ��ġ�� Condition Vector�� �߰�(Concatenation)

    * Generator�� �Է� ( z, fake_label)

    * Dircriminator�� �Է� 

      E[ log(D(real_x)) ] : (_img, real_label) , gt_real
      
      E[ log(1-D(G(z))) ] : ( generated_img, fake_label) ,gt_fake


< �˾Ƶα� >

 - ���� ���� ��, one hot vector�� ���� ��ȯ�Ͽ�, noise sample data �� concatenation 

   => one hot vector�� ���� ��ȯ �Ͽ��� ������ �Ű���� ���� ��ȯ ������ �н� ���� �ʱ� ������ ���� �����ս��� ���� ������ ������ �߻�

   => ���� ��ȯ�� �ƴ�, embedding layer�� ���� label�� ���� embedding vector�� ���� ����ϴ� ������ �ذ� (2022-10-15)


 - noise sample data z�� ���Ͽ� real label concatenation 

   => z ���� �����ϰ� ��������� ���̱� ������, ���� real label�� concatenation �� �ʿ� ���� np.random.randint(0, 10, batch_size)�� ���� �����ϰ� ���� (2022-10-15)

      label�� batch_size��ŭ ���� �����ϱ� ������ 60000 % batch_size == 0 �� �ǵ��� batch_size�� �����ϴ� ���� ����


 - ���� ���ϴ� �̹����� �����ϰ� �ʹٸ� NN�� depth�� �÷��� �ϴ� ������ ����

   Ư�� ���ڸ� �Է¹޾� condition vector�� Ȱ���ϸ� �̹��� ������ �� ���� �ʴ� ������ �־��� 

   ������ ã�´ٸ� �� �� ������Ʈ �ϱ� 
   

'''

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torchvision.transforms as trans

from cGAN_model import Generator, Discriminator

import argparse
import numpy as np

## set params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default = 0.0002, help = "Adam : learning rate")
parser.add_argument("--n_epochs", type = int, default = 40, help = "number of epoch of training")
parser.add_argument("--batch_size", type = int, default = 32, help = "size of the batches")
parser.add_argument("--num_class", type = int, default = 10, help = "number of MNIST dataset's class")
parser.add_argument("--latent_dim", type = int, default = 100 , help = "latent vector z's dimension")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--sample_interval", type = int, default = 2500 , help = "interval between image sampling")
parser.add_argument("--img_size", type = int , default = 28, help = "size of MNIST image ")
parser.add_argument("--target_batch", type = int, default = 10, help = "size of custom data's batch")
opt = parser.parse_args()

## load data
train_data = MNIST(
    root = './data',
    train = True,
    download = True,
    transform = trans.Compose( [trans.Resize(opt.img_size) , trans.ToTensor() , trans.Normalize([0.5], [0.5]) ] ),
    )
# 60,000 / 32 = 1875
train_loader = torch.utils.data.DataLoader(train_data, batch_size = opt.batch_size, shuffle = True)

## set models
discriminator = Discriminator(opt.img_size, opt.num_class).to(device)
generator = Generator(opt.latent_dim, opt.num_class).to(device)

## set loss & optimizer
criterion = nn.MSELoss()
optimizer_g = optim.Adam(generator.parameters(), lr = opt.lr, betas=(opt.b1, opt.b2) )
optimizer_d = optim.Adam(discriminator.parameters(), lr =opt.lr, betas=(opt.b1, opt.b2) )


def chk_img(batch_done):
    
    n_row = 10 

    z = torch.randn(n_row ** 2, opt.latent_dim).to(device)

    # [ 0, 0, 0,  ... 1,1,1, ... , 9, 9,9]
    label = torch.LongTensor( [ num for num in range(n_row) for _ in range(n_row)  ] ).to(device)

    gen_imgs = generator( z, label)

    save_image(gen_imgs, f'./cGAN/train_output/{batch_done}.png', nrow = n_row, normalize = True )

## training
for epoch in range(opt.n_epochs):

    for batch_idx, (img, label) in enumerate(train_loader):

        # set img(real), noise_img(z) , label(real), fake_label       
        img = torch.FloatTensor(img).to(device)
        z = torch.randn(img.shape[0], opt.latent_dim).to(device)

        label = torch.LongTensor(label).to(device)
        fake_label = torch.LongTensor(np.random.randint(0, opt.num_class, label.shape[0])).to(device)

        # ground truth
        gt_real = torch.cuda.FloatTensor(img.shape[0], 1).fill_(1.0)
        gt_fake = torch.cuda.FloatTensor(img.shape[0], 1).fill_(0.0)

        # =============
        # Generator traning
        # =============
        optimizer_g.zero_grad()

        gen_output = generator(z , fake_label )

        g_loss = criterion( discriminator( gen_output, fake_label) , gt_real)
        g_loss.backward()
        optimizer_g.step()


        # =============
        # Discriminator traning
        # =============
        optimizer_d.zero_grad()

        real_loss = criterion( discriminator(img, label) , gt_real)
        fake_loss = criterion( discriminator(gen_output.detach(), fake_label), gt_fake)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_d.step()


        # len(train_loader) = 60,000 /32 = 1875 (iterations) , epcoh = 0 ~ 200
        # ������ ���������� ����� Ȯ���� ���� interval ���� 
        batch_done = epoch * len(train_loader) + batch_idx

        if batch_done % opt.sample_interval == 0 :
            
            chk_img(batch_done)
            
    print(f'Epoch : {epoch}/{opt.n_epochs}  |  G loss : {g_loss}  | D loss : {d_loss}')

    if epoch == (opt.n_epochs)-1 : 
        torch.save(generator.state_dict(), "./cGAN/cGAN.pt")

