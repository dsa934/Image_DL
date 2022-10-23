
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-21

'''

import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, input_dim = 3):
        
        super().__init__()

        def dis_block(input_dim, output_dim, normalize= True):

            layers = [ nn.Conv2d(input_dim, output_dim, kernel_size = 4, stride = 2, padding = 1 )] 

            if normalize :

                layers.append(nn.InstanceNorm2d(output_dim))

            # nn.LeakyReLU(negative_slope, ... )
            layers.append(nn.LeakyReLU(0.2, inplace = True) )

            return layers

        self.model = nn.Sequential(
            # [3, 256, 256] => [64, 128, 128]
            *dis_block(input_dim, 64, normalize = False ),
            # [64, 128, 128] => [128, 64, 64]
            *dis_block(64,128),
            # [128, 64, 64] => [256, 32, 32]
            *dis_block(128,256),
            # [256, 32, 32] => [512, 16, 16]
            *dis_block(256,512),
            # [512, 16, 16 ] => [512, 17, 17]
            nn.ZeroPad2d((1, 0, 1, 0)),
            # [512, 17, 17] => [1, 16, 16]
            nn.Conv2d(512, 1, kernel_size = 4, stride = 1 , padding = 1 )
            
            )

    def forward(self, img):

        # init img dim = [batch, 3, 128, 128]
        
        return self.model(img)


class Residual_Block(nn.Module):

    def __init__(self, input_dim):

        super().__init__()
        
        self.model = nn.Sequential(

            # init data = [256, 32, 32]

            # [256, 32, 32] => [256, 34, 34]
            nn.ReflectionPad2d(1),
            # [256, 34, 34] => [256, 32, 32]
            nn.Conv2d(input_dim, input_dim, kernel_size = 3),
            nn.InstanceNorm2d(input_dim),
            nn.ReLU(inplace = True),
            
            # [256, 32, 32] => [256, 34, 34]
            nn.ReflectionPad2d(1),
            # [256, 34, 34] => [256, 32, 32]
            nn.Conv2d(input_dim, input_dim, kernel_size = 3),
            nn.InstanceNorm2d(input_dim)
            )

    def forward(self, data):
        # data_dim == self.block(data)_dim         
        return data + self.model(data)


class Generator(nn.Module):

    
    def __init__(self, input_dim = 3):
        
        super().__init__()

        self.model = nn.Sequential(

            # init  data = [3, 256, 256]

            # ================
            # First Conv layers
            # ================
            # [3, 256, 256] -> [ 3, 256+6, 256+6 ]
            nn.ReflectionPad2d(input_dim),
            # [3, 262, 262] -> [64,256,256]
            nn.Conv2d(input_dim, 64, kernel_size = 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace = True),

            # ================
            # Second Conv layers ( Down Sampling)
            # ================
            # [64, 256, 256] -> [128, 128, 128]
            nn.Conv2d(64, 128, kernel_size = 3 , stride = 2, padding = 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace = True),

            # [128, 128, 128] -> [256, 64, 64]
            nn.Conv2d(128, 256, kernel_size = 3 , stride = 2, padding = 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace = True),

            # ================
            # 6 Residual Blocks
            # ================
            # [256, 64, 64] -> [256, 64, 64]
            Residual_Block(256),
            Residual_Block(256),
            Residual_Block(256),
            Residual_Block(256),
            Residual_Block(256),
            Residual_Block(256),
            Residual_Block(256),
            Residual_Block(256),
            Residual_Block(256),

            # ================
            # Thrid Conv layers ( Up Sampling )
            # ================            
            # [256, 64, 64] -> [128, 128, 128]
            nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace = True),

            # [128, 128, 128] -> [64, 256, 256]
            nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace = True),

            # ================
            # Last Conv layers ( RGB feature map)
            # ================
            # [64, 256, 256] -> [64, 262, 262]            
            nn.ReflectionPad2d(input_dim),
            # [64, 262, 262] -> [3, 256, 256]
            nn.Conv2d(64, 3, kernel_size = 7 ),
            nn.Tanh(),
            
            )

    def forward(self, data):

        return self.model(data)



