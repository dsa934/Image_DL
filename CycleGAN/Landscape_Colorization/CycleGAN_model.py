
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-18


< Model Structure >

 - 2 GAN models ( 2 generatos, 2 discriminators )

 
 * Generator

   - 3 Conv layers + 6 Residual Networks + 1 Conv layer

     * 3 Conv layers     : init , downsampling , upsampling  (학습이 잘 되지 않는 경우 down / up sampling layer가 추가 될 수 있음 )

     * 6 Residual block  : 채널 수 , 해상도 유지 

     * 1 Conv layers     : RGB 에 대한 Feature map 형성


 * Discriminator

   - Patch Discriminator 

      => input img shape = [128,128]을 전부 판별 하는것이 아닌 일부(patch)만 [8, 8] 판별
       
         [128, 128] => ... => [8,8]   depth = 4 


< Residual Block >

 - Channel의 수는 유지 

 - 최초 입력                      : (H,W)

   ReflectionPad2d                : (H+2, W+2)

   conv2d(... , kernel= 3 , .. )  : ( H + 2  - 2 , W + 2 - 2 )

   결과적으로 채널 수, 이미지 크기 모두 유지 되기 떄문에 x + self.block(x) 연산의 차원 충돌은 발생하지 않는다.


< 알아두면 유용한 method & info >

 1. nn.ReflectionPad2d() , input tensor dim = [N, C, H, W]

    => nn.ReflectionPad2d(left, ight, top, bottom)

       after ReflectionPad2d = [N, C, H + top + bottom , W + left + right ]

    => nn.ReflectionPad2d(alpha)

       after ReflectionPad2d = [N , C, H + 2alpha , W + 2alpha]

       
 2. nn.LeakyReLU(negative_slope, in_place = True)

    => in_place : 덮어쓰기 , negative_slope : 음수 쪽 기울기 정도


< 논문에서 서술된 Network 구조  >

Network Architecture We adopt the architecture for our generative networks from Johnson et al. [23] who have shown impressive results for neural style transfer and superresolution. 
This network contains three convolutions, several residual blocks [18], two fractionally-strided convolutions with stride 1/2, and one convolution that maps features to RGB. We use 6 blocks for 128 × 128 images and 9
blocks for 256×256 and higher-resolution training images. Similar to Johnson et al. [23], we use instance normalization [53]. For the discriminator networks we use 70 × 70 PatchGANs [22, 30, 29], which aim to classify whether
70 × 70 overlapping image patches are real or fake. Such a patch-level discriminator architecture has fewer parameters than a full-image discriminator and can work on arbitra


=> 3 conv layers + several residual blocks + 2 Convtranspos + 1 convolution that maps features to RGB.
=> if input_img = [128,128] then 6 residual blocks used

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
            # [3, 128, 128] => [64, 64, 64]
            *dis_block(input_dim, 64),
            # [64, 64, 64] => [128, 32, 32]
            *dis_block(64,128),
            # [128, 32, 32] => [256, 16, 16]
            *dis_block(128,256),
            # [256, 16, 16] => [512, 8, 8]
            *dis_block(256,512),
            # [512,8,8] => [1,8,8]
            nn.Conv2d(512, 1, kernel_size = 3, stride = 1 , padding = 1 )
            
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

            # init  data = [3, 128, 128]

            # ================
            # First Conv layers
            # ================
            # [3, 128, 128] -> [ 3, 128+6, 128+6 ]
            nn.ReflectionPad2d(input_dim),
            # [3, 134, 134] -> [64,128,128]
            nn.Conv2d(input_dim, 64, kernel_size = 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace = True),

            # ================
            # Second Conv layers ( Down Sampling)
            # ================
            # [64, 128, 128] -> [128, 64, 64]
            nn.Conv2d(64, 128, kernel_size = 3 , stride = 2, padding = 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace = True),

            # [128, 64, 64] -> [256, 32, 32]
            nn.Conv2d(128, 256, kernel_size = 3 , stride = 2, padding = 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace = True),

            # ================
            # 6 Residual Blocks
            # ================
            # [256, 32, 32] -> [256, 32, 32]
            Residual_Block(256),
            Residual_Block(256),
            Residual_Block(256),
            Residual_Block(256),
            Residual_Block(256),
            Residual_Block(256),

            # ================
            # Thrid Conv layers ( Up Sampling )
            # ================            
            # [256, 32, 32] -> [128, 64, 64]
            nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace = True),

            # [128, 64, 64] -> [64, 128, 128]
            nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace = True),

            # ================
            # Last Conv layers ( RGB feature map)
            # ================
            # [64, 128, 128] -> [64, 134, 134]            
            nn.ReflectionPad2d(input_dim),
            # [64, 134, 134] -> [3, 128, 128]
            nn.Conv2d(64, 3, kernel_size = 7 ),
            nn.Tanh(),
            
            )

    def forward(self, data):

        return self.model(data)



