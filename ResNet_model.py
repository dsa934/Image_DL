# -*- coding: cp949 -*-
'''
 Author / date : Jinwoo Lee / 2022-10-04

< Characteristic of ResNet > 

 - 1���� init block , 4���� Residual block ���� 

 - 1���� Residual block�� 2���� conv layer�� ���� 

   * first conv       : feature map ũ�� ����

   * second conv      : feature map ũ�� ����

   * skip connections : second conv ���� ���� 

   * basic order      : Conv -> BN -> Activation function 


< Feature map size ��� >

  - next width( height ���� ) =   ( W - FW + 2P )               ( W : Current width, FW : filter Width, P : padding , S : stride )
                                   �ѤѤѤѤѤѤ�  + 1 
                                         S
   
  - if filter(kernel) size = 3 x 3 and stride = 1 �̸�  �� ���Ŀ� ���� convolution layer�� ��� �Ͽ��� feature map�� ũ�Ⱑ ������ ���� 


  - filter map flow  :   1 -> 64 -> 64 -> 64 -> 128 -> 128 -> 256 -> 256 -> 512 -> 512 -> num_class(e.g MNIST = 10)


< Data dimension flow >

[128, 1, 28, 28]     : init data = [batch, channel, width, height]

[128, 64, 14, 14 ]   : after init block 
----------------------------------------
[128, 64, 14, 14 ]   : block 1 start
 
[128, 64, 7, 7 ]     : block 1 end 
----------------------------------------
[128, 64, 7, 7 ]     : block 2 start

[128, 128, 4, 4 ]    : block 2 end
----------------------------------------
[128, 128, 4, 4 ]    : block 3 start

[128, 256, 2, 2 ]    : block 3 end
----------------------------------------
[128, 256, 2, 2 ]    : block 4 start

[128, 512, 1, 1 ]    : block 4 end
----------------------------------------

'''

import torch.nn as nn 
import torch.nn.functional as F

class Residual_block(nn.Module):

    def __init__(self, input_dim, output_dim):

        super().__init__()

        self.rb_conv1 = nn.Conv2d(input_dim, input_dim, kernel_size = 3 , stride = 1, padding = 1, bias = False)
        self.rb_bn1   = nn.BatchNorm2d(input_dim)

        self.rb_conv2 = nn.Conv2d(input_dim, output_dim, kernel_size = 3 , stride = 2, padding = 1 , bias = False)
        self.rb_bn2   = nn.BatchNorm2d(output_dim)


        self.skip_connec = nn.Sequential(

            nn.Conv2d(input_dim, output_dim, kernel_size = 3, stride = 2, padding = 1 , bias = False ),
            nn.BatchNorm2d(output_dim)
            
            )

    def forward(self, x):
        
        for_skip_x = x 

        x = F.relu ( self.rb_bn1( self.rb_conv1(x) ) )


        x =  F.relu( self.rb_bn2( self.rb_conv2(x) ) + self.skip_connec(for_skip_x)  )

        return x



class ResNet(nn.Module):

    def __init__(self, num_class):

        super().__init__()

        # Init block  ( without Skip connections ) 
        self.init_conv = nn.Conv2d(1, 64, kernel_size = 3 , stride = 2, padding = 1, bias = False)
        self.init_bn = nn.BatchNorm2d(64)
        
        # ResNet with Skip connections
        
        self.block_1 = Residual_block(64, 64)       
        
        self.block_2 = Residual_block(64, 128)
        
        self.block_3 = Residual_block(128, 256)
        
        self.block_4 = Residual_block(256, 512)
        
        # output
        self.output = nn.Linear(512, num_class)
        
    def forward(self,x):
        
        # init x = [128,1,28,28]

        # after init block x = [ 128, 64, 14, 14]
        x = F.relu( self.init_bn( self.init_conv(x) )  )
        
        # after block 1, x = [128, 64, 7, 7]
        x = self.block_1(x)
        
        # after block 2, x = [128, 128, 4, 4]
        x = self.block_2(x)
        
        # after block 3, x = [128, 256, 2, 2]
        x = self.block_3(x)
        
        # after block 4, x = [128, 512, 1, 1]
        x = self.block_4(x)
        
        # x = [128, 512]
        x = x.view(x.shape[0], -1)

        # x = [128, 10]
        _output = self.output(x)

        return _output
        