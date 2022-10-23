
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-19


< Cycle GAN 복원 >

 - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (ICCV 2017)

 - Horse2Zebra datasets

   => image size = [256,256] 임으로 network size 증가( 9 residual blocks) 


< transform.Compose >

- transform.Resize( int(opt.img_size * 1.12) ) : 이미지 크기를 좀 더 크게 증가


 ** monet2photo, horse2zebra dataset의 경우 landscape colorization dataset 보다 이미지의 크기가 크고, 장수가 많기 떄문에 

   * 만족스러운 결과를 얻기 위한 최소 epoch 증가 ( 150~!60 )

   * h/w 적 제한으로 인해 batch_size = 1 로 설정해야 함으로 상대적 시간이 오래 걸림 

     이로 인해 실제 학습은 25 ~ 35 epoch 밖예 진행을 못해보았으나 , 여러 자료를 검색한 결과 코드 구현 부분에서 잘못된 점은 찾지 못했음 


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

from Horse2Zebra_CycleGAN_utils import ImageDataset, Buffer, weights_init_normal, LambdaLR
from Horse2Zebra_CycleGAN_model import Generator, Discriminator


import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type = int , default = 170, help = "num of epochs")
parser.add_argument('--img_size', type = int, default = 256, help="size of transformed image")
parser.add_argument('--lr', type = float, default = 0.0005, help="Adam : learning rate")
parser.add_argument('--batch_size', type = int, default = 1, help="size of the batches")
parser.add_argument('--test_batch_size', type = int , default = 4, help = "size of the val batches")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--lambda_cycle', type = int , default = 10 , help = "Cycle loss weight parameter")
parser.add_argument('--lambda_identity', type = int, default = 5, help = "Identity loss weight parameter")
parser.add_argument("--sample_interval", type = int, default = 1500 , help = "interval between image sampling")
parser.add_argument('--data_path', type = str, default ='./CycleGAN/dataset/horse2zebra_dataset', help = " ")
parser.add_argument('--model_save_path', type = str, default ='./CycleGAN/Horse2Zebra/', help = " best performance model save points")
parser.add_argument('--result_save_path', type = str, default ='./CycleGAN/Horse2Zebra/result', help = " best performance model save points")
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## set data format
data_transform = trans.Compose(
    [ 
      # image 크기 조금 키우기
      trans.Resize( (opt.img_size, opt.img_size), trans.InterpolationMode.BICUBIC),
      trans.RandomCrop((opt.img_size, opt.img_size)),
      # 좌우반전 -> pix2pix 처럼 pair 이미지가 아닌 단일 이미지로 존재하기 떄문에 가능 
      trans.RandomHorizontalFlip(),
      trans.ToTensor(),
      trans.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )   ])

## data load
train_data = ImageDataset(opt.data_path, data_transform,  'train')
test_data = ImageDataset(opt.data_path, data_transform, 'test')

# train_loader = (1334 or 1067) / 1 = ~, test_loader = (120 or 140 ) / 4 = 30 ~ 33
train_loader = DataLoader(train_data, batch_size = opt.batch_size , shuffle = True)
test_loader = DataLoader(test_data, batch_size = opt.test_batch_size, shuffle = True)


## set model
# G_horse2zebra := G() in paper , G_zebra2horse := F() in paper
G_horse2zebra = Generator().to(device)
G_zebra2horse = Generator().to(device)

D_horse = Discriminator().to(device)
D_zebra = Discriminator().to(device)

## init model weight
G_horse2zebra.apply(weights_init_normal)
G_zebra2horse.apply(weights_init_normal)

D_horse.apply(weights_init_normal)
D_zebra.apply(weights_init_normal)

## set loss
criterion_GAN = nn.MSELoss().to(device)
criterion_Cycle = nn.L1Loss().to(device)
criterion_Identity = nn.L1Loss().to(device)

## set optimizer
G_optimizer = optim.Adam(itertools.chain(G_zebra2horse.parameters() , G_horse2zebra.parameters() ) , lr = opt.lr, betas = (opt.b1, opt.b2) )
horse_optimizer = optim.Adam(D_horse.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2) )
zebra_optimizer = optim.Adam(D_zebra.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2) )


## set buffer
horse_buffer = Buffer()
zebra_buffer = Buffer()

## Training
for epoch in range(opt.n_epochs):

    for batch_idx, data in enumerate(train_loader):

        img_horse , img_zebra = data['horse'].to(device), data['zebra'].to(device)

        # ground truth 
        # dim = discriminator's output dim
        gt_real = torch.FloatTensor(img_horse.shape[0] , 1, 16, 16).fill_(1.0).to(device)
        gt_fake = torch.FloatTensor(img_horse.shape[0], 1, 16, 16).fill_(0.0).to(device)


        img_horse2zebra = G_horse2zebra(img_horse) 
        img_zebra2horse = G_zebra2horse(img_zebra)

        # ============
        # Training - Generator
        # ============
        G_optimizer.zero_grad()

        # GAN Loss
        # after generator , image shape = [6, 3, 256, 256]
        # after discriminator , image shape = gt_real = [batch, 1, 16, 16]
        loss_horse2zebra = criterion_GAN( D_zebra( img_horse2zebra ) , gt_real ) 
        loss_zebra2horse = criterion_GAN( D_horse( img_zebra2horse ) , gt_real )
        loss_GAN = (loss_horse2zebra + loss_zebra2horse ) /2

        # Cycle Loss
        c_loss_horse    = criterion_Cycle( G_zebra2horse (  img_horse2zebra  ) , img_horse ) 
        c_loss_zebra = criterion_Cycle( G_horse2zebra ( img_zebra2horse ), img_zebra)
        loss_Cycle = ( c_loss_horse + c_loss_zebra) / 2

        # Identity Loss
        i_loss_horse = criterion_Identity( G_zebra2horse(img_horse), img_horse)
        i_loss_zebra = criterion_Identity( G_horse2zebra(img_zebra) , img_zebra)
        loss_Identity = (i_loss_horse + i_loss_zebra ) /2

        # final loss
        g_loss = loss_GAN + ( opt.lambda_cycle * loss_Cycle) + ( opt.lambda_identity * loss_Identity)

        g_loss.backward()
        G_optimizer.step()


        # ============
        # Training - Discriminator horse
        # ============
        horse_optimizer.zero_grad()

        horse_real_loss = criterion_GAN( D_horse(img_horse) , gt_real)

        buffer_out_img_zebra2horse = horse_buffer.push_pop(img_zebra2horse)

        horse_fake_loss = criterion_GAN( D_horse( buffer_out_img_zebra2horse.detach() ) , gt_fake ) 

        D_horse_loss = (horse_real_loss + horse_fake_loss) / 2
        
        D_horse_loss.backward()
        horse_optimizer.step()
        

        # ============
        # Training - Discriminator zebra
        # ============
        zebra_optimizer.zero_grad()

        zebra_real_loss = criterion_GAN( D_zebra(img_zebra) , gt_real) 
        
        buffer_out_img_horse2zebra = zebra_buffer.push_pop(img_horse2zebra)

        zebra_fake_loss = criterion_GAN( D_zebra( buffer_out_img_horse2zebra.detach() ), gt_fake)

        D_zebra_loss = (zebra_real_loss + zebra_fake_loss) / 2

        D_zebra_loss.backward()
        zebra_optimizer.step()


        done = epoch * len(train_loader) + batch_idx

        if done % opt.sample_interval == 0 :

            G_horse2zebra.eval()
            G_zebra2horse.eval()

            test_img = next(iter(test_loader))

            test_horse , test_zebra = test_img['horse'].to(device), test_img['zebra'].to(device)

            test_horse2zebra = G_horse2zebra(test_horse)
            test_zebra2horse = G_zebra2horse(test_zebra)

            # x축을 따라 각각의 그리디 이미지 생성 
            test_horse = make_grid(test_horse, nrow = 4, normalize = True)
            test_zebra  = make_grid(test_zebra, nrow = 4, normalize= True)
            test_horse2zebra  = make_grid(test_horse2zebra, nrow = 4, normalize= True)
            test_zebra2horse  = make_grid(test_zebra2horse, nrow = 4, normalize= True)
            
            image_grid = torch.cat( (test_horse, test_horse2zebra, test_zebra, test_zebra2horse) , 1)
            save_image(image_grid, f'{opt.result_save_path}/{done}.png', Normalize = False )

        if batch_idx % 500 == 0 : print(f'iterations : {batch_idx} / {len(train_loader)}')
    print(f'Epoch : {epoch}/{opt.n_epochs}  |  G loss : {g_loss}  |  D_horse loss : {D_horse_loss}  |  D_zebra loss : {D_zebra_loss}')
    

    ## save model
    if epoch == opt.n_epochs-1 :

        torch.save(G_horse2zebra.state_dict(), opt.model_save_path +'G_horse2zebra.pt' )
        torch.save(G_zebra2horse.state_dict(), opt.model_save_path + 'G_zebra2horse' )
        torch.save(D_horse.state_dict(), opt.model_save_path  + 'D_horse.pt')
        torch.save(D_zebra.state_dict(), opt.model_save_path  + 'D_zebra.pt')


        print("saved model")
