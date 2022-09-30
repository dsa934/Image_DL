
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-27


 < Dimension >

  * Generator

    => [128,input_dim] �� noise vector z   =>   MNIST Img [128, 1, 28 , 28](���, ����,����) 

       CNN�� ������� �ʱ� ������, [1,100] �� 1D vector�� [1,784]�� 1D vector�� ��ȯ ��

       [1,784] -> [ 1,28,28] �� ����  ��, 1D vector���� 2D img data�� ��ȯ  (batch �� ����)



 < �˾Ƶθ� ������ method & info >

  1. detach()

     =>  ������ tensor���� gradient ���İ� ���� �ʴ� �ټ� ���� ( <-> Clone() : ���� tensor�� �״�� ���� )

         1. Generator�� �н��ϱ� ���ؼ��� Discriminator�� �����ؾ� �ϰ�
        
            -> generator�� ���� �н� ������ ������� 

         2. discriminator�� �н��ϱ� ���ؼ��� Generator�� �����ؾ� �Ѵ� 

            -> discriminator �н��� , Generator�� ���� ������ _generated_img�� �ʿ��ϸ�, 

               update ������ _generated_img�� �ʿ�������, discriminator update�� ���� 
            
               gradient ���İ� ���� �ʴ� _generated_img�� �ʿ��� 


  2. congiguous()

     => narrow(), view(), expand(), transpose() ���� �Լ��� ���ο� Tensor�� �����ϴ°� x 

        ������ Tensor���� ��Ÿ �����͸� �����Ͽ� �츮���� ������ ���� -> �޸� �󿡼� ���� ������ ���� 

        ��, ����������� �޸𸮿� �÷��� ������ �߿��� ��� ���ϴ� ����� ������ ���� �� ������, 

        ������ ���ڰ� ����ϴ� ������ �����ϱ� ���ؼ� congiguous�� ����Ͽ� ������ ���� �� �� ����



  3. �̹��� �����Ϳ� ���� Ȯ�� ����

     => img data�� ������ Ư¡ ������ �� ������ ǥ�� ���� (�ٺ��� Ȯ������)

        100 x 100 img�� 10,000 ���� ������ �ϳ��� �� 

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
            # in_place : ������ ������ν� ��� ���⵵ ����
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