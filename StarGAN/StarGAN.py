
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-31


< StarGAN 복원 >

 - StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation (2018 CVPR)

 - 해당 논문은 datasets 으로 CelebA , RaFD 데이터셋을 활용 

   * CelebA : 40개의 속성

   * RaFD   : 8개의 속성 

   따라서, train, test part 구현은 하되, 성능 확인은 학습된 모델을 load 하여 사용 


< 알아 두기 >

 * numel()

    => Returns the total number of elements in the input tensor.  

       print ( torch.numel( torch.zeros(4,4) ) ) = 16 


 * clamp_(0,1)

    => Clamps all elements in input into the range [ min, max ]. 


 * transform.Resize( [alpha, beta ] )

   => alpha, beta 를 동시에 선언해야 각각 row(alpha), column(beta)에 맞게 크기 변환

      alpha 하나만 선언할 경우, img의 column만 변화 

   
 * Google Colab 

   => 코랩안쓰고 코딩시

      plt.imshow(image)
      plt.show() # 이거 안하면 이미지 안뜸 혼동 ㄴㄴ


'''
import os
# PIL 오류 방지 
os.environ['KMP_DUPLICATE_LIB_OK']='True' 

import torch
import torch.nn as nn
import torch.optim as optim

from StarGAN_model import Generator
from StarGAN_model import Discriminator

import numpy as np
from torchvision.utils import save_image


## SET arguments
import argparse
parser = argparse.ArgumentParser()

# data params
parser.add_argument('--img_size', type = int, default = 256, help="size of transformed image")
parser.add_argument('--dataset', type = str, default = 'CelebA' , help ="")
parser.add_argument('--data_property', type = list, default = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"], help=" Select the properties of the image you want to create")
parser.add_argument('--custom_img_path', type = str , default = './StarGAN/custom_data' , help = "")

# model params
parser.add_argument('--model_load_path', type = str, default ='./StarGAN/model_save', help = " load best performance modelpoints")
parser.add_argument('--result_save_path', type = str, default ='./StarGAN/result', help = " save starGAN's result with Custom data")
parser.add_argument('--c_dim', type = int, default = 5 , help = "condition vector dimension with CelebA dataset")
parser.add_argument('--c2_dim', type = int, default = 8 , help = "condition vector dimension with RaFD datasets")
parser.add_argument('--g_conv_dim', type = int, default = 64  , help = "# of Conv filter of generator")
parser.add_argument('--d_conv_dim', type = int , default = 64 , help = "# of Conv filter of discriminator")
parser.add_argument('--g_repeat_num', type = int , default = 6 , help = "# of residual block in generator")
parser.add_argument('--d_repeat_num', type = int , default = 6 , help = "# of residual block in discriminator")

# train params
parser.add_argument('--g_lr', type = float, default = 0.0001, help="Adam : Geneator's learning rate")
parser.add_argument('--d_lr', type = float, default = 0.0001, help="Adam : Discriminator's learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

# test params
parser.add_argument('--test_iter', type = int, default = 200000, help = "It brings the parameters of the model trained n times.")
opt = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## Set Solver 
class Solver(object):

    def __init__(self, opt):

        # model params
        self.c_dim = opt.c_dim
        self.c2_dim = opt.c2_dim
        self.img_size = opt.img_size

        self.g_conv_dim = opt.g_conv_dim
        self.d_conv_dim = opt.d_conv_dim
        self.g_repeat_num = opt.g_repeat_num
        self.d_repeat_num = opt.d_repeat_num

        # train & data params
        self.dataset = opt.dataset
        self.g_lr = opt.g_lr
        self.d_lr = opt.d_lr
        self.b1 = opt.b1
        self.b2 = opt.b2
        self.property = opt.data_property

        # test params
        self.test_iter = opt.test_iter

        # set models
        self.make_model()

    def make_model(self):

        if self.dataset in ['CelebA' , 'RaFD'] : 

            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num).to(device)
            self.D = Discriminator(self.img_size, self.d_conv_dim, self.c_dim, self.d_repeat_num).to(device)


        elif self.dataset in ['Both'] :

            # +2 : for mask vector
            self.G = Generator(self.g_conv_dim, self.c_dim + self.c2_dim + 2 , self.g_repeat_num).to(device)
            self.D = Discriminator(self.img_size, self.d_conv_dim, self.c_dim + self.c2_dim, self.d_repeat_num).to(device)


        self.g_optimizer = optim.Adam(self.G.parameters(), self.g_lr, [self.b1, self.b2])
        self.d_optimizer = optim.Adam(self.D.parameters(), self.d_lr, [self.b1, self.b2])

        self.show_network(self.G, 'Generator')
        self.show_network(self.D, 'Discriminator')


    def show_network(self, model, name):

        num_params = 0 

        for p in model.parameters(): num_params += p.numel()
        print(f"model : {model}")
        print(f"name : {name}\t # of parameters :{num_params}")


    def denorm(self, data):
        # convert range [-1,1] to [0,1]
        output = (data+1) / 2 
        return output.clamp_(0, 1)

    def load_model(self, best_iter_point):

        print(f"Load the parameters when the model's performance is highest : {best_iter_point}" )

        self.G.load_state_dict( torch.load( opt.model_load_path + f'/{best_iter_point}-G.ckpt') ) 
        self.D.load_state_dict( torch.load( opt.model_load_path + f'/{best_iter_point}-D.ckpt') ) 



## load well-trained model params
translation = Solver(opt)
translation.load_model(opt.test_iter)

## Custom image testing 
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision import transforms as trans

data_transform = trans.Compose([
    trans.Resize([opt.img_size, opt.img_size]),
    trans.ToTensor(),
    trans.Normalize( (0.5, 0.5, 0.5) , (0.5, 0.5, 0.5) )
    ])

custom_img_path = opt.custom_img_path + "/WS.jpg"

image = Image.open(custom_img_path)

# image = [1,3,307,256] =>  크기 [307,256]의 컬러 이미지(3 channels) 1장을 뜻하기 위해 unsqueeze 추가 
image = data_transform(image).unsqueeze(0).to(device)

## ploting
# subplot(row, col, index)
plt.subplot(1, 4, 1)
plt.imshow(translation.denorm(image.data.cpu()).squeeze(0).permute(1,2,0))
plt.title("Original img data")

# [Black_Hair, Blond_Hair, Brown_Hair, Male, Young]
c_trg = [ [0, 1, 0, 1, 1] ]

plt.subplot(1,4,2)
c_trg = torch.FloatTensor(c_trg).to(device)
output = translation.G(image, c_trg)
plt.imshow(translation.denorm(output.data.cpu()).squeeze(0).permute(1, 2, 0))
plt.title(" Blond + male + yong")

plt.subplot(1,4,3)
c_trg = [ [0, 1, 0, 1, 0] ]
c_trg = torch.FloatTensor(c_trg).to(device)
output = translation.G(image, c_trg)
plt.imshow(translation.denorm(output.data.cpu()).squeeze(0).permute(1, 2, 0))
plt.title(" Blond + male ")

plt.subplot(1,4,4)
c_trg = [ [0, 0, 1, 0, 1] ]
c_trg = torch.FloatTensor(c_trg).to(device)
output = translation.G(image, c_trg)
plt.imshow(translation.denorm(output.data.cpu()).squeeze(0).permute(1, 2, 0))
plt.title(" Brown hair + female")

plt.savefig(opt.result_save_path + '/ws')
plt.show()