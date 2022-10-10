
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-11


 < Dimension >

  * Generator

    => 기존의 GAN과 동일 하지만, Condition Vector를 noise sampling vector와 concatenation 

       condition vector dim = [ 128, num_class ] 임으로  ( MNIST 라서 num_class = 10 )

       Generator의 입력 차원은 self.input_dim + num_class(10)


  * Discriminator 

    => 28 x 28 image를 flatten 하게 펼침 ( CNN을 사용하지 않기 떄문)

       여기에 condition vector가 추가 됨으로 

       입력 dim = ( 1 * 28 * 28 ) + 10 
 

'''

import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm1d


class Generator(nn.Module):
    
    def __init__(self, input_dim):

        super().__init__()

        self.input_dim = input_dim


        self.model = nn.Sequential(

            nn.Linear(self.input_dim, 128),
            # batchnorm1d ( num_feattures, eps, momentum)
            # eps      : a value added to the denominator for numerical stability
            # momentum : the value used for the running_mean and running_var computation
            nn.BatchNorm1d(128, momentum = 0.8),
            
            # LeakyRelu(negative slope, inplace)  
            # in_place : 변수를 덮어씀으로써 계산 복잡도 감소
            nn.LeakyReLU(0.2 , inplace= True),

            nn.Linear(128,256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2 , inplace= True),

            nn.Linear(256,512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2 , inplace= True),

            nn.Linear(512,1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2 , inplace= True),

            nn.Linear(1024, 1 * 28 * 28 ),
            nn.Tanh()

            )


    def forward(self, x):
        
        # init x_dim = [batch, latent_input_dim] = [128,100 + 10 ]
        
        _batch = x.shape[0] 

        x = self.model(x)

        gener_output = x.view(_batch, 1, 28, 28)

        return gener_output


class Discriminator(nn.Module):
    

    def __init__(self):
        
        super().__init__()


        self.model = nn.Sequential(

            nn.Linear((1 * 28 * 28 )+ 10, 1024),
            nn.LeakyReLU(0.2 , inplace = True ),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2 , inplace = True ),

            nn.Linear(512,256),
            nn.LeakyReLU(0.2 , inplace = True ),

            nn.Linear(256,1),
            nn.Sigmoid()
            
            )


    def forward(self, x, c):

        # x init dim = [128,1,28,28]
        # c init dim = [128, 10]
        _batch = x.shape[0]

        # flatten_x = [128, 784]
        flatten_x = x.view(_batch, -1)

        # x + conditional vector
        conditional_x = torch.cat( (flatten_x, c), dim = 1)

        dis_output = self.model(conditional_x)

        return dis_output