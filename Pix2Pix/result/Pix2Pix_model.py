
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-14


< Model Structure >

 - Contract Block  : Conv2d -> Norm -> ReLU -> Dropout 

 
 - Expand Block    : ConvTranspose2d -> Norm -> ReLU -> Dropout


 - Generator(U-net 구조 사용 )

   => Encoder ( 8 Contract Block ) + Decoder ( 7 Expanding Block + last layers )
   
      256 x 256 크기 이미지의 크기를 2배씩 감소시키면서 1 x 1 까지 도달 

      [256, 256] -> [128, 128] -> ... [1,1]  : 8번의 Contract Block 필요 

      [1,1] -> [2,2] -> ... -> [128,128]     : 7번의 Expand Block 필요 

      
      * last layers : expanding block 의 경우 contract block의 feature map을 concatenation 해야 하지만

                      마지막 layer의 경우 skip connections이 필요가 없기 떄문에 conv2dTranspose 단독으로 사용하기 때문에 

                      decoder 파트에서는 7 expanding block + last block으로 구성 


 - Discriminator (중요)

   => Convolutional PatchGAN 분류 모델 사용 

      discriminator의 output = [batch, 512, 16, 16] 으로 return 

      [256,256] 이미지 전체를 판별하는 것이 아닌 일부(patch)에 대한 True/False 판별

      이로 인해 적은 수의 params , 학습 속도 향상, 이미지의 크기에 대한 부담이 덜 함
     

< Method >

 - nn.InstanceNorm2d  vs nn.BatchNorm2d

   => 전자 : 각 Batch 내에서, instance 기준으로 Normalization 

      후자 : Whole dataset 기준, Batch에 대한 normalization

'''
import torch
import torch.nn as nn
from torch.nn.modules.activation import Tanh 

class Contract_Block(nn.Module):
    
    def __init__(self, input_dim, output_dim, normalize = True , dropout = False):

        super().__init__()

        layers = [ nn.Conv2d(input_dim, output_dim, kernel_size = 4, stride = 2, padding = 1 , bias = False) ]

        if normalize : 
            
            layers.append(nn.InstanceNorm2d(output_dim) ) 

        layers.append( nn.ReLU(inplace = True) ) 

        if dropout : 

            layers.append( nn.Dropout2d(dropout))

        self.ct_block = nn.Sequential(*layers)


    def forward(self, x):
        
        return self.ct_block(x)



class Expand_Block(nn.Module):
    
    def __init__(self, input_dim , output_dim, dropout = False ):

        super().__init__()

        layers = [ nn.ConvTranspose2d(input_dim , output_dim, kernel_size = 4, stride = 2 , padding = 1, bias = False ) ]

        layers.append( nn.InstanceNorm2d(output_dim) )
        layers.append( nn.ReLU( inplace = True ) )

        if dropout : 

            layers.append( nn.Dropout(dropout) ) 

        self.ed_block = nn.Sequential(*layers)

    def forward(self, x, contract_x):
        
        expand_x = self.ed_block(x)

        concate_x = torch.cat( (expand_x, contract_x) , dim = 1 )

        return concate_x 

class Encoder(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        # [3, 256, 256] -> [64, 128, 128]
        self.enc1 = Contract_Block(input_dim, 64)
        
        # [64, 128, 128] -> [128, 64, 64]
        self.enc2 = Contract_Block(64, 128)

        # [128, 64, 64] -> [256, 32, 32]
        self.enc3 = Contract_Block(128, 256)

        # [256, 32, 32] -> [512, 16, 16]
        self.enc4 = Contract_Block(256, 512)

        # [512, 16, 16] -> [512, 8, 8]
        self.enc5 = Contract_Block(512, 512)
        
        # [512, 8, 8] -> [512, 4, 4]
        self.enc6 = Contract_Block(512, 512)
        
        # [512, 4, 4] -> [512, 2, 2]
        self.enc7 = Contract_Block(512, 512)
        
        # [512, 2, 2] -> [512, 1, 1]
        self.enc8 = Contract_Block(512, 512)
        

    def forward(self, x):

        # init x : [10, 3, 256, 256]
        enc_1 = self.enc1(x)
        enc_2 = self.enc2(enc_1)
        enc_3 = self.enc3(enc_2)
        enc_4 = self.enc4(enc_3)
        enc_5 = self.enc5(enc_4)
        enc_6 = self.enc6(enc_5)
        enc_7 = self.enc7(enc_6)
        enc_8 = self.enc8(enc_7)
        return enc_1, enc_2, enc_3, enc_4, enc_5, enc_6, enc_7, enc_8


class Decoder(nn.Module):

    def __init__(self, output_dim):

        super().__init__()

        # [512, 1, 1] -> [512 * 2, 2, 2]
        self.dec1 = Expand_Block(512, 512)
                                               
        # [512 * 2 , 2, 2] -> [512, 4, 4] -> [ 512 * 2 , 4, 4] , concatenation
        self.dec2 = Expand_Block(1024, 512)
        
        # [512 * 2 , 4, 4] -> [512, 8, 8] -> [ 512 * 2 , 8, 8] , concatenation
        self.dec3 = Expand_Block(1024, 512)
        
        # [512 * 2 , 8, 8] -> [512, 16, 16] -> [ 512 * 2 , 16, 16] , concatenation
        self.dec4 = Expand_Block(1024, 512)
        
        # [512 * 2 , 16, 16] -> [256, 32, 32] -> [ 256 * 2 , 32, 32] , concatenation
        self.dec5 = Expand_Block(1024, 256)
        
        # [256 * 2 , 32, 32] -> [128, 64, 64] -> [ 128 * 2, 64, 64] , concatenation
        self.dec6 = Expand_Block(512, 128)
        
        # [128 * 2 , 64, 64] -> [64, 128, 128] -> [ 64 * 2, 128, 128] , concatenation
        self.dec7 = Expand_Block(256, 64)
        
        self.last = nn.Sequential(

            nn.ConvTranspose2d(128, output_dim, kernel_size = 4, stride = 2 , padding = 1, bias = False),
            nn.Tanh(),
            
            )

    def forward(self, enc_1, enc_2, enc_3, enc_4, enc_5, enc_6, enc_7, enc_8):

        dec_1 = self.dec1(enc_8, enc_7)
        dec_2 = self.dec2(dec_1, enc_6)
        dec_3 = self.dec3(dec_2, enc_5)
        dec_4 = self.dec4(dec_3, enc_4)
        dec_5 = self.dec5(dec_4, enc_3)
        dec_6 = self.dec6(dec_5, enc_2)
        dec_7 = self.dec7(dec_6, enc_1)
        
        decoder_output = self.last(dec_7)

        return decoder_output



class Generator(nn.Module):
    
    def __init__(self, input_dim = 3 , output_dim = 3 ):
        
        super().__init__()

        self.encoder = Encoder(input_dim)
        self.decoder = Decoder(output_dim)

    def forward(self,x):
        
        enc_1, enc_2, enc_3, enc_4, enc_5, enc_6, enc_7, enc_8 = self.encoder(x)

        output = self.decoder(enc_1, enc_2, enc_3, enc_4, enc_5, enc_6, enc_7, enc_8)

        return output

class Dis_Block(nn.Module):

    def __init__(self, input_dim, output_dim, normalize = True ):

        super().__init__()

        layers = [ nn.Conv2d(input_dim, output_dim, kernel_size = 4, stride = 2 , padding =1)] 

        if normalize :

            layers.append(nn.InstanceNorm2d(output_dim) )

        layers.append( nn.LeakyReLU(0.2 , inplace = True ) )
        

        self.dis_block = nn.Sequential(*layers)

    def forward(self, x):
        
        return self.dis_block(x)


class Discriminator(nn.Module):

    def __init__(self, input_dim = 3):

        super().__init__()

        self.dis_model = nn.Sequential(

            # [3 * 2 , 256, 256] -> [64, 128, 128]
            Dis_Block(input_dim * 2 , 64, normalize = False ),

            # [64, 128, 128] -> [128, 64, 64]
            Dis_Block(64, 128),

            # [128, 64, 64] -> [256, 32, 32]
            Dis_Block(128, 256),

            # [256, 32, 32] -> [512, 16, 16]
            Dis_Block(256, 512),

            # [512, 16, 16] -> [1, 16, 16]
            nn.Conv2d(512, 1, kernel_size = 3, stride = 1, padding = 1, bias = False ),
            
            )


    def forward(self, x_img, condition_img):
        
        # input_img dim = [10, 6, 256, 256]
        input_img = torch.cat( (x_img, condition_img) , dim = 1)
        
        return self.dis_model(input_img)





