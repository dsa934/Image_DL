
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-21


< Cycle GAN 복원 >

 - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (ICCV 2017)
 
 - Monet to Photo datasets

'''

import os

import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision.transforms as trans
from torchvision.utils import save_image, make_grid

from CycleGAN_Monet2Photo_utils import ImageDataset, Buffer
from CycleGAN_Monet2Photo_model import Generator, Discriminator


import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type = int , default = 200, help = "num of epochs")
parser.add_argument('--img_size', type = int, default = 256, help="size of transformed image")
parser.add_argument('--lr', type = float, default = 0.0005, help="Adam : learning rate")
parser.add_argument('--batch_size', type = int, default = 1, help="size of the batches")
parser.add_argument('--test_batch_size', type = int , default = 4, help = "size of the val batches")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--lambda_cycle', type = int , default = 10 , help = "Cycle loss weight parameter")
parser.add_argument('--lambda_identity', type = int, default = 5, help = "Identity loss weight parameter")
parser.add_argument("--sample_interval", type = int, default = 3000 , help = "interval between image sampling")
parser.add_argument('--data_path', type = str, default ='./CycleGAN/dataset/monet2photo_dataset', help = " ")
parser.add_argument('--model_save_path', type = str, default ='./CycleGAN/Monet2Photo/', help = " best performance model save points")
parser.add_argument('--result_save_path', type = str, default ='./CycleGAN/Monet2Photo/result', help = " best performance model save points")
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## set data format
data_transform = trans.Compose(
    [ 
      # image 크기 조금 키우기
      trans.Resize( int(opt.img_size * 1.12 ), trans.InterpolationMode.BICUBIC),
      trans.RandomCrop((opt.img_size, opt.img_size)),
      # 좌우반전 -> pix2pix 처럼 pair 이미지가 아닌 단일 이미지로 존재하기 떄문에 가능 
      trans.RandomHorizontalFlip(),
      trans.ToTensor(),
      trans.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )   ])

## data load
train_data = ImageDataset(opt.data_path, data_transform,  'train')
test_data = ImageDataset(opt.data_path, data_transform, 'test')

train_loader = DataLoader(train_data, batch_size = opt.batch_size , shuffle = True)
test_loader = DataLoader(test_data, batch_size = opt.test_batch_size, shuffle = True)

## set model
# G_monet2photo := G() in paper , G_photo2monet := F() in paper
# monet(bw) , photo(rgb)
G_monet2photo = Generator().to(device)
G_photo2monet = Generator().to(device)

D_monet = Discriminator().to(device)
D_photo = Discriminator().to(device)

## set loss
criterion_GAN = nn.MSELoss().to(device)
criterion_Cycle = nn.L1Loss().to(device)
criterion_Identity = nn.L1Loss().to(device)

## set optimizer
G_optimizer = optim.Adam(itertools.chain(G_photo2monet.parameters() , G_monet2photo.parameters() ) , lr = opt.lr, betas = (opt.b1, opt.b2) )
monet_optimizer = optim.Adam(D_monet.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2) )
photo_optimizer = optim.Adam(D_photo.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2) )


## set buffer
monet_buffer = Buffer()
photo_buffer = Buffer()

## Training
for epoch in range(opt.n_epochs):

    for batch_idx, data in enumerate(train_loader):

        img_photo , img_monet = data['photo'].to(device), data['monet'].to(device)

        # ground truth 
        # dim = discriminator's output dim
        gt_real = torch.FloatTensor(img_photo.shape[0] , 1, 16, 16).fill_(1.0).to(device)
        gt_fake = torch.FloatTensor(img_photo.shape[0], 1, 16, 16).fill_(0.0).to(device)


        img_monet2photo = G_monet2photo(img_monet) 
        img_photo2monet = G_photo2monet(img_photo)

        # ============
        # Training - Generator
        # ============
        G_optimizer.zero_grad()

        # GAN Loss
        # after generator , image shape = [6, 3, 128, 128]
        # after discriminator , image shape = gt_real = [batch, 1, 8, 8]
        loss_monet2photo = criterion_GAN( D_photo( img_monet2photo ) , gt_real ) 
        loss_photo2monet = criterion_GAN( D_monet( img_photo2monet ) , gt_real )
        loss_GAN = (loss_monet2photo + loss_photo2monet ) /2

        # Cycle Loss
        c_loss_monet    = criterion_Cycle( G_photo2monet (  img_monet2photo  ) , img_monet ) 
        c_loss_photo = criterion_Cycle( G_monet2photo ( img_photo2monet ), img_photo)
        loss_Cycle = ( c_loss_monet + c_loss_photo) / 2

        # Identity Loss
        i_loss_monet = criterion_Identity( G_photo2monet(img_monet), img_monet)
        i_loss_photo = criterion_Identity( G_monet2photo(img_photo) , img_photo)
        loss_Identity = (i_loss_monet + i_loss_photo ) /2

        # final loss
        g_loss = loss_GAN + ( opt.lambda_cycle * loss_Cycle) + ( opt.lambda_identity * loss_Identity)

        g_loss.backward()
        G_optimizer.step()


        # ============
        # Training - Discriminator monet
        # ============
        monet_optimizer.zero_grad()

        monet_real_loss = criterion_GAN( D_monet(img_monet) , gt_real)

        buffer_out_img_photo2monet = monet_buffer.push_pop(img_photo2monet)

        monet_fake_loss = criterion_GAN( D_monet( buffer_out_img_photo2monet.detach() ) , gt_fake ) 

        D_monet_loss = (monet_real_loss + monet_fake_loss) / 2
        
        D_monet_loss.backward()
        monet_optimizer.step()
        

        # ============
        # Training - Discriminator photo
        # ============
        photo_optimizer.zero_grad()

        photo_real_loss = criterion_GAN( D_photo(img_photo) , gt_real) 
        
        buffer_out_img_monet2photo = photo_buffer.push_pop(img_monet2photo)

        photo_fake_loss = criterion_GAN( D_photo( buffer_out_img_monet2photo.detach() ), gt_fake)

        D_photo_loss = (photo_real_loss + photo_fake_loss) / 2

        D_photo_loss.backward()
        photo_optimizer.step()


        done = epoch * len(train_loader) + batch_idx

        if done % opt.sample_interval == 0 :

            G_monet2photo.eval()
            G_photo2monet.eval()

            test_img = next(iter(test_loader))

            test_photo , test_monet = test_img['photo'].to(device), test_img['monet'].to(device)

            test_monet2photo = G_monet2photo(test_monet)
            test_photo2monet = G_photo2monet(test_photo)

            # x축을 따라 각각의 그리디 이미지 생성 
            test_photo = make_grid(test_photo, nrow = 4, normalize = True)
            test_monet  = make_grid(test_monet, nrow = 4, normalize= True)
            test_monet2photo  = make_grid(test_monet2photo, nrow = 4, normalize= True)
            test_photo2monet  = make_grid(test_photo2monet, nrow = 4, normalize= True)
            
            image_grid = torch.cat( (test_monet, test_monet2photo, test_photo, test_photo2monet) , 1)
            save_image(image_grid, f'{opt.result_save_path}/{done}.png', Normalize = False )

        if batch_idx % 500 == 0 : print(f'iterations : {batch_idx} / {len(train_loader)}')
    print(f'Epoch : {epoch}/{opt.n_epochs}  |  G loss : {g_loss}  |  D_monet loss : {D_monet_loss}  |  D_photo loss : {D_photo_loss}')
    

    ## save model
    if epoch == opt.n_epochs-1 :

        torch.save(G_monet2photo.state_dict(), opt.model_save_path +'G_monet2photo.pt' )
        torch.save(G_photo2monet.state_dict(), opt.model_save_path + 'G_photo2monet' )
        torch.save(D_monet.state_dict(), opt.model_save_path  + 'D_monet.pt')
        torch.save(D_photo.state_dict(), opt.model_save_path  + 'D_photo.pt')


        print("saved model")
