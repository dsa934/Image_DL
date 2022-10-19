
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-18


< Cycle GAN ���� >

 - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (ICCV 2017)
 
 - Landscape Colorizaiton datasets


< Generator Loss Fucntion >

 - GAN Loss (MSE Loss)

   => loss_bw2color = criterion_GAN( D_rgb( img_bw2rgb ) , gt_real ) 

      �Է� ������        : ��� �̹���

      ��� geneartor     : G_bw2color

      ��� discriminator : D_color

      �� ��� �̹����� �Է����� ���� �̹����� ���� ��, ���� discriminator�� �ӿ��� �Ѵ� ( �׷����� ���� �̹��� ����, �Է� �����Ϳ� �������� ���� )


   => loss_color2bw = criterion_GAN( D_bw( img_rgb2bw ) , gt_real )

      �Է� ������        : ���� �̹���

      ��� geneartor     : G_color2bw

      ��� discriminator : D_bw

      �� ���� �̹����� �Է����� ��� �̹����� ���� ��, ��� discriminator�� �ӿ��� �Ѵ� ( �׷����� ��� �̹��� ����, �Է� �����Ϳ� �������� ���� )


 - Cycle Loss (L1 Loss)
 
   => c_loss_bw  = criterion_Cycle ( G_rgb2bw ( G_bw2rgb(img_gray)) , img_gray )

      �Է� ������    : ��� �̹���

      ��� generator :   G_bw2rgb -> G_rgb2bw

      ��, ��� �̹����� ���� �̹����� �����, ������� ���� �̹����� �ٽ� ��� �̹����� ���� , ��� �̹����� �� 
 
   => c_loss_color = criterion_Cycle( G_bw2rgb ( img_rgb2bw ), img_color)

      ���� �ݴ�� 



 - Identity Loss (L1 Loss)

   => ���� ������ ���� �ؾ��ϴ� ��쿡 ��� 

      i_loss_bw = criterion_Identity( G_rgb2bw(img_gray), img_gray)

      i_loss_color = criterion_Identity( G_bw2rgb(img_color) , img_color)

      ��, ��� �̹��� �����⿡ ��� �̹����� �ְ�, ��� �̹����� �� �Ͽ� ���̰� ���������� (�ڱ� �ڽŰ� �� , identity loss)

          Į�� �̹��� �����⿡ Į�� �̹����� �ְ�, Į�� �̹����� �� �Ͽ� ���̰� ���������� (�ڱ� �ڽŰ� �� , identity loss)



< BW discriminator  Loss Fucntion >

 - GAN loss (MSE Loss)

   => Real loss :  d_bw_real_loss = criterion_GAN( D_bw(img_gray) , gt_real)       

                   ��� �̹����� ��� �з��⿡ �־����� ���� 1�� �ǵ���

   => Fake loss :  d_bw_fake_loss = criterion_GAN( D_bw( buffer_out_img_rgb2bw ) , gt_fake ) 

                   ���� �̹����� rgb2bw�� �־� ���� gray image�� buffer�� �ְ�, 

                   buffer���� ��µ� gray image�� bw_discriminator�� ���� ��� ���� 
                  
                   gt_fake�� ���� GAN loss ��� 



< color discriminator  Loss Fucntion >

 - GAN loss (MSE Loss)

   => Real loss :  d_rgb_real_loss = criterion_GAN( D_rgb(img_color) , gt_real)       

                   ���� �̹����� ���� �з��⿡ �־����� ���� 1 �� �ǵ���

   => Fake loss :  d_rgb_fake_loss = criterion_GAN( D_rgb( buffer_out_img_bw2rgb ) , gt_fake ) 

                   ��� �̹����� bw2rgb�� �־� ���� color image�� buffer�� �ְ�, 

                   buffer���� ��µ� colorimage�� rgb_discriminator�� ���� ��� ���� 
                   
                   gt_fake�� ���� GAN loss ��� 




< �˾Ƶθ� ������ method & info >

 1. optimizer = optim.Adam (  itertools.chain( model1.parameters() , model2.parameters() ) , lr = opt.lr , betas = (opt.b1 , opt.b2 ) ) 
 
    => itertools.chain( contents ) : contents ���� �ϳ��� �ڷ����·� ���´�

       ��, �� model�� �Ķ���͵��� �ϳ��� ��� optimizer�� �Ű������� ��� 


 2. next(iter(train_loader))

    =>  x = 'example'   current x.type = str
     
        x = iter(x)     current x.type = str_iterator 

        iter : iterable(�ݺ�����) �� ��ü�� iter �Լ��� ���� iterator ��ü�� ��ȯ 

        iterator ��ü : �ѹ��� �ϳ��� �ش� ��ü�� ��Ҹ� ������� access ���� ( for���� ����������, �̴� for���� ���� �������� ���� -> ����)

                        ������� �����͸� ������ �� �ش� �����͸� ����ϱ� ������, �޸� ������ ������ ��Ը� ������ ó������ ����


        next()  : itertor ��ü�� �����͸� ���������� access 



'''

import os

from matplotlib.colors import Normalize
import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision.transforms as trans
from torchvision.utils import save_image, make_grid

from CycleGAN_utils import ImageDataset, Buffer
from CycleGAN_model import Generator, Discriminator


import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--raw_img_size', type = int , default = 150, help = "size of the raw image")
parser.add_argument('--n_epochs', type = int , default = 30, help = "num of epochs")
parser.add_argument('--img_size', type = int, default = 128, help="size of transformed image")
parser.add_argument('--lr', type = float, default = 2e-4, help="Adam : learning rate")
parser.add_argument('--batch_size', type = int, default = 6, help="size of the batches")
parser.add_argument('--val_batch_size', type = int , default = 4, help = "size of the val batches")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--lambda_cycle', type = int , default = 10 , help = "Cycle loss weight parameter")
parser.add_argument('--lambda_identity', type = int, default = 5, help = "Identity loss weight parameter")
parser.add_argument("--sample_interval", type = int, default = 1500 , help = "interval between image sampling")
parser.add_argument('--data_path', type = str, default ='./CycleGAN/dataset/landscape_dataset', help = " ")
parser.add_argument('--model_save_path', type = str, default ='./CycleGAN/Landscape_Colorization', help = " best performance model save points")
parser.add_argument('--result_save_path', type = str, default ='./CycleGAN/Landscape_Colorization/result', help = " best performance model save points")
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## set data format
data_transform = trans.Compose(
    [ trans.Resize( ( opt.img_size, opt.img_size), trans.InterpolationMode.BICUBIC),
      trans.ToTensor(),
      trans.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )   ])

## data load
train_data = ImageDataset(opt.data_path, data_transform,  'train')
val_data = ImageDataset(opt.data_path, data_transform, 'val')

# train_loader = 8,300 / 6 = 1384, val_loader = 20
train_loader = DataLoader(train_data, batch_size = opt.batch_size , shuffle = True)
val_loader = DataLoader(val_data, batch_size = opt.val_batch_size, shuffle = True)

## set model
# G_bw2rgb := G() in paper , G_rgb2bw := F() in paper
G_bw2rgb = Generator().to(device)
G_rgb2bw = Generator().to(device)

D_bw = Discriminator().to(device)
D_rgb = Discriminator().to(device)

## set loss
criterion_GAN = nn.MSELoss().to(device)
criterion_Cycle = nn.L1Loss().to(device)
criterion_Identity = nn.L1Loss().to(device)

## set optimizer
G_optimizer = optim.Adam(itertools.chain(G_rgb2bw.parameters() , G_bw2rgb.parameters() ) , lr = opt.lr, betas = (opt.b1, opt.b2) )
bw_optimizer = optim.Adam(D_bw.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2) )
rgb_optimizer = optim.Adam(D_rgb.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2) )


## set buffer
bw_buffer = Buffer()
rgb_buffer = Buffer()

## Training
for epoch in range(opt.n_epochs):

    for batch_idx, data in enumerate(train_loader):

        img_color , img_gray = data['color'].to(device), data['gray'].to(device)

        # ground truth 
        # dim = discriminator's output dim
        gt_real = torch.FloatTensor(img_color.shape[0] , 1, 8, 8).fill_(1.0).to(device)
        gt_fake = torch.FloatTensor(img_color.shape[0], 1, 8, 8).fill_(0.0).to(device)


        img_bw2rgb = G_bw2rgb(img_gray) 
        img_rgb2bw = G_rgb2bw(img_color)

        # ============
        # Training - Generator
        # ============
        G_optimizer.zero_grad()

        # GAN Loss
        # after generator , image shape = [6, 3, 128, 128]
        # after discriminator , image shape = gt_real = [batch, 1, 8, 8]
        loss_bw2color = criterion_GAN( D_rgb( img_bw2rgb ) , gt_real ) 
        loss_color2bw = criterion_GAN( D_bw( img_rgb2bw ) , gt_real )
        loss_GAN = (loss_bw2color + loss_color2bw ) /2

        # Cycle Loss
        c_loss_bw    = criterion_Cycle( G_rgb2bw (  img_bw2rgb  ) , img_gray ) 
        c_loss_color = criterion_Cycle( G_bw2rgb ( img_rgb2bw ), img_color)
        loss_Cycle = ( c_loss_bw + c_loss_color) / 2

        # Identity Loss
        i_loss_bw = criterion_Identity( G_rgb2bw(img_gray), img_gray)
        i_loss_color = criterion_Identity( G_bw2rgb(img_color) , img_color)
        loss_Identity = (i_loss_bw + i_loss_color ) /2

        # final loss
        g_loss = loss_GAN + ( opt.lambda_cycle * loss_Cycle) + ( opt.lambda_identity * loss_Identity)

        g_loss.backward()
        G_optimizer.step()


        # ============
        # Training - Discriminator bw
        # ============
        bw_optimizer.zero_grad()

        bw_real_loss = criterion_GAN( D_bw(img_gray) , gt_real)

        buffer_out_img_rgb2bw = bw_buffer.push_pop(img_rgb2bw)

        bw_fake_loss = criterion_GAN( D_bw( buffer_out_img_rgb2bw.detach() ) , gt_fake ) 

        D_bw_loss = (bw_real_loss + bw_fake_loss) / 2
        
        D_bw_loss.backward()
        bw_optimizer.step()
        

        # ============
        # Training - Discriminator color
        # ============
        rgb_optimizer.zero_grad()

        rgb_real_loss = criterion_GAN( D_rgb(img_color) , gt_real) 
        
        buffer_out_img_bw2rgb = rgb_buffer.push_pop(img_bw2rgb)

        rgb_fake_loss = criterion_GAN( D_rgb( buffer_out_img_bw2rgb.detach() ), gt_fake)

        D_rgb_loss = (rgb_real_loss + rgb_fake_loss) / 2

        D_rgb_loss.backward()
        rgb_optimizer.step()


        done = epoch * len(train_loader) + batch_idx

        if done % opt.sample_interval == 0 :

            G_bw2rgb.eval()
            G_rgb2bw.eval()

            val_img = next(iter(val_loader))

            val_color , val_gray = val_img['color'].to(device), val_img['gray'].to(device)

            val_bw2rgb = G_bw2rgb(val_gray)
            val_rgb2bw = G_rgb2bw(val_color)

            # x���� ���� ������ �׸��� �̹��� ���� 
            val_color = make_grid(val_color, nrow = 4, normalize = True)
            val_gray  = make_grid(val_gray, nrow = 4, normalize= True)
            val_bw2rgb  = make_grid(val_bw2rgb, nrow = 4, normalize= True)
            val_rgb2bw  = make_grid(val_rgb2bw, nrow = 4, normalize= True)
            
            image_grid = torch.cat( (val_gray, val_bw2rgb, val_color, val_rgb2bw) , 1)
            save_image(image_grid, f'{opt.result_save_path}/{done}.png', Normalize = True )

        if batch_idx % 500 == 0 : print(f'iterations : {batch_idx} / {len(train_loader)}')
    print(f'Epoch : {epoch}/{opt.n_epochs}  |  G loss : {g_loss}  |  D_bw loss : {D_bw_loss}  |  D_rgb loss : {D_rgb_loss}')
    

    ## save model
    if epoch == opt.n_epochs-1 :

        torch.save(G_bw2rgb.state_dict(), opt.model_save_path +'G_bw2rgb.pt' )
        torch.save(G_rgb2bw.state_dict(), opt.model_save_path + 'G_rgb2bw' )
        torch.save(D_bw.state_dict(), opt.model_save_path  + 'D_bw.pt')
        torch.save(D_rgb.state_dict(), opt.model_save_path  + 'D_rgb.pt')


        print("saved model")
