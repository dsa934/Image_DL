
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-19


< 논문에서 서술된 Network 구조  >

Network Architecture We adopt the architecture for our generative networks from Johnson et al. [23] who have shown impressive results for neural style transfer and superresolution. 
This network contains three convolutions, several residual blocks [18], two fractionally-strided convolutions with stride 1/2, and one convolution that maps features to RGB. We use 6 blocks for 128 × 128 images and 9
blocks for 256×256 and higher-resolution training images. Similar to Johnson et al. [23], we use instance normalization [53]. For the discriminator networks we use 70 × 70 PatchGANs [22, 30, 29], which aim to classify whether
70 × 70 overlapping image patches are real or fake. Such a patch-level discriminator architecture has fewer parameters than a full-image discriminator and can work on arbitra


=> 3 conv layers + several residual blocks + 2 Convtranspos + 1 convolution that maps features to RGB.
=> if input_img = [256,256] then 9 residual blocks used
=> Appendix 7.2 


< Discriminator > 

 - 70 x 70 PathGAN 

   => discriminator 차원 변화 : 256(raw input) -> 128(first conv) -> 64(second conv) -> 32(thrid conv) -> 16(fourth conv) -> 17(padding) - > 16(last conv)

      Receptive Field = 70 x 70  ( Receptive Field : 하나의 뉴런이 원본 이미지에서 담당하는 범위 )

      Formula :  ( output_size - 1 ) * stride + kernel_size 

      Calculate RF  :  last feature map = 16 x 16 , 1개의 뉴런에 대한  RF 계산 


                       frist step  :  ( 1 - 1 ) * 1 + 4  = 4

                       second step :  ( 4 - 1 ) * 1 + 4  = 7     (10)

                       thrid tstep :  ( 7 - 1 ) * 2 + 4  = 16    (22)

                       fourth step :  ( 16 - 1 ) * 2 + 4 = 34    (46)

                       last step   :  ( 34 -1 ) * 2 + 4  = 70    (94)
 

    ** 현재 구현된 코드 상의 discriminator에 대한 RF는 94 이다. 

       논문에서는 70 x 70 patch GAN을 의미했지만, 코드 압축을 위하여 ( 마지막 conv layer를 제외하고, 4개의 dis_block 으로 묶어서 표현하기 위해 )

       조금 94 x 94 pathGAN으로 설계하였다.

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



