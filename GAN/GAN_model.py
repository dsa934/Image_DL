
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-27


 < Dimension >

  * Generator

    => [128,input_dim] 의 noise vector z   =>   MNIST Img [128, 1, 28 , 28](흑백, 가로,세로) 

       CNN을 사용하지 않기 때문에, [1,100] 의 1D vector를 [1,784]의 1D vector로 변환 후

       [1,784] -> [ 1,28,28] 의 형태  즉, 1D vector에서 2D img data로 변환  (batch 는 생략)



 < 알아두면 유용한 method & info >

  1. detach()

     =>  기존의 tensor에서 gradient 전파가 되지 않는 텐서 생성 ( <-> Clone() : 기존 tensor를 그대로 복사 )

         1. Generator를 학습하기 위해서는 Discriminator를 고정해야 하고
        
            -> generator가 먼저 학습 됨으로 상관없음 

         2. discriminator를 학습하기 위해서는 Generator를 고정해야 한다 

            -> discriminator 학습에 , Generator에 의해 생성된 _generated_img가 필요하며, 

               update 이전의 _generated_img가 필요함으로, discriminator update를 위한 
            
               gradient 전파가 되지 않는 _generated_img가 필요함 


  2. congiguous()

     => narrow(), view(), expand(), transpose() 등의 함수는 새로운 Tensor를 형성하는것 x 

        기존의 Tensor에서 메타 데이터만 수정하여 우리에게 정보를 제공 -> 메모리 상에서 같은 공간을 공유 

        즉, 연산과정에서 메모리에 올려진 순서가 중요할 경우 원하는 결과가 나오지 않을 수 있으며, 

        실제로 저자가 기대하는 순서를 유지하기 위해서 congiguous를 사용하여 에러를 방지 할 수 있음



  3. 이미지 데이터에 대한 확를 분포

     => img data는 다차원 특징 공간의 한 점으로 표현 가능 (다변수 확률분포)

        100 x 100 img는 10,000 차원 공간의 하나의 점 

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
        
        # init x_dim = [batch, latent_input_dim] = [128,100]
        
        _batch = x.shape[0] 

        x = self.model(x)

        gener_output = x.view(_batch, 1, 28, 28)

        return gener_output


class Discriminator(nn.Module):
    

    def __init__(self):
        
        super().__init__()


        self.model = nn.Sequential(

            nn.Linear(1 * 28 * 28, 1024),
            nn.LeakyReLU(0.2 , inplace = True ),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2 , inplace = True ),

            nn.Linear(512,256),
            nn.LeakyReLU(0.2 , inplace = True ),

            nn.Linear(256,1),
            nn.Sigmoid()
            
            )


    def forward(self, x):

        # x init dim = [128,1,28,28]
        _batch = x.shape[0]

        # flatten_x = [128, 784]
        flatten_x = x.view(_batch, -1)

        dis_output = self.model(flatten_x)

        return dis_output