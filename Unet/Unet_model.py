
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-06

< Data dimension flow >

[batch, 1, 512, 512]      : init data = [batch, channel, width, height]

[batch, 64, 512, 512]     : 채널 수만 변함

[batch, 64, 512, 512]     : 채널 수 유지 

[batch, 64, 254, 254]     :  2 x 2 max pool

   (생략)

[batch, num_class, height, width] :  Semantic segmentation task 임으로, 픽셀 단위 분류작업이 필요 하며,

                                     마지막 layer 이 후의 채널 수 = 분류 해야하는 클래스의 수 



 *  pool & unpool 

   - 수축 경로(Contracting Path) 에서 feature map의 크기를 줄이는 것은 maxpooling(size = n) 의 역할

   - 확장 경로(Expanding Path) 에서 feature map의 크기를 늘리는 것은 ConvTranspose2D( ... , stride = n ) 의 역할

     따라서 ConvTranspose2D에서 다뤄지는 입출력 채널 수는 동일하게 유지 



 * nn.Conv2d( ... , padding = 1 ) 

   - 원본 논문에서는 padding = 0 으로 줬기 때문에, Convolutional layer를 거쳐 갈 때마다 image_size = [ H-2, W-2] 

     이미지의 사이즈가 감소하기 때문에 Contracting , Expadning path 간 skip connection 진행 시 

     Contracting path에서 생성된 feature map의 크기가 Expanding path에서 다뤄지는 feature map의 크기보다 커서 그대로 concatenation은 불가능

     Contracting path의 feature map을 일정 크기 잘라낸 버전을 사용 함

     그러나 해당 코드에서는 padding =1 로 줌으로써 Contracting path ~ Expanding path 간의 픽셀 크기의 변화는 pooling layer 에서만 이루어짐 ( Expanding path 에서는 concatenation으로 이미지 크기 증가 )

     따라서 그대로 Contracting path's feature map 과 Expanding Path's feature map에 대한 concatenation 가능 

     논문 복원을 100% 그대로 진행할 경우 Cropping 에 대한 부분을 추가적으로 고려할 필요성 존재 


< Connection > 

 - 동일 계층의 encoder, decoder 성분이 서로 concatenation 

 - [batch, channel, height, width ] 임으로 channel 기준으로 concatenation 

   => dec_unpool4 = [batch, 512, height, width ]  

      enc4_2      = [batch, 512,  height, width ]

 - Contracting path 

   * first conv layer  : 채널의 수 증가

   * Second conv layer : 채널의 수 유지
 
 - Expanding path

   * first conv layer  : 채널의 수 감소

     => encoder와 decoder 가 데칼코마니 구조라고 생각 했을때, 채널의 수가 유지되어야 하지만 

        Unet은 encoder / decoder layer 각각의 output이 concatenation 되는 과정을 거치기 떄문에

        decoder의 첫번쨰 conv layer는 concate 된 입력 데이터를 처리 함으로, 채널의 수가 2배로 늘어난다

        따라서 원래는 유지가 맞지만, concated data를 처리함으로 채널의 수를 감소시키는 것으로 보이는 것이다.


   * Second conv layer : 채널의 수 감소 


'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Unet_Block(nn.Module):

    def __init__(self, input_dim, output_dim):
        
        super().__init__()
        
        self.conv = nn.Conv2d(input_dim , output_dim , kernel_size = 3, padding = 1, stride =1 , bias = True )

        self.bn   = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        
        x = F.relu ( self.bn( self.conv(x) ) )

        return x 

class Unet(nn.Module):

    def __init__(self, num_class):

        super().__init__()
        
        # Contracting path (encoder)
        self.enc1_1 = Unet_Block(1,64)
        self.enc1_2 = Unet_Block(64,64)
        self.enc_pool1 = nn.MaxPool2d(kernel_size = 2)

        self.enc2_1 = Unet_Block(64,128)
        self.enc2_2 = Unet_Block(128,128)
        self.enc_pool2 = nn.MaxPool2d(kernel_size = 2)

        self.enc3_1 = Unet_Block(128,256)
        self.enc3_2 = Unet_Block(256,256)
        self.enc_pool3 = nn.MaxPool2d(kernel_size = 2)

        self.enc4_1 = Unet_Block(256,512)
        self.enc4_2 = Unet_Block(512,512)
        self.enc_pool4 = nn.MaxPool2d(kernel_size = 2)

        self.enc5_1 = Unet_Block(512,1024)

        # Expanding path (decoder)
        self.dec5_1 = Unet_Block(1024,512)

        self.dec_unpool4 = nn.ConvTranspose2d(512, 512, kernel_size = 2, stride = 2, padding = 0, bias = True)
        self.dec4_1 = Unet_Block(2 * 512 , 512)  # concatenation\
        self.dec4_2 = Unet_Block(512, 256)

        self.dec_unpool3 = nn.ConvTranspose2d(256, 256, kernel_size = 2, stride = 2, padding = 0, bias = True)
        self.dec3_1 = Unet_Block(2 * 256, 256)
        self.dec3_2 = Unet_Block(256, 128)

        self.dec_unpool2 = nn.ConvTranspose2d(128, 128, kernel_size = 2, stride = 2, padding = 0, bias = True)
        self.dec2_1 = Unet_Block(2 * 128, 128)
        self.dec2_2 = Unet_Block(128, 64)

        self.dec_unpool1 = nn.ConvTranspose2d(64, 64, kernel_size = 2, stride = 2, padding = 0, bias = True)
        self.dec1_1 = Unet_Block(2 * 64, 64)
        self.dec1_2 = Unet_Block(64, 64)


        '''
         논문에서는 마지막 layer에 의해 생성되는 feature map의 개수가 num_class 와 동일 (해당 example 에서는 2 )

         그러나 2가지 이유로 복원 과정에서 1로 설정 함 

         1) binary classification problem 임으로  num_class = 2 설정 + nn.CrossEntropyLoss() 를 사용하는 것보다 

            num_class = 1 로 설정 후 , nn.BCEWithLogitsLoss() 를 사용해도 문제 되지 않음 

            * nn.BCEWithLogitsLoss()  = sigmoid activation f + BCE Loss  


         2)  RuntimeError: Expected object of scalar type Long but got scalar type Float for argument #2 'target' in call to _thnn_nll_loss2d_forward

             Model의 output 이 activation function을 거치지 않기 떄문에 label과의 data type이 맞지 않아 loss 계산에 문제가 될 수 있음 
        '''
        self.last = nn.Conv2d(64, num_class, kernel_size = 1, stride = 1 , padding = 0, bias = True)
        

    def forward(self, x):
        
        enc1_1    = self.enc1_1(x)     
        enc1_2    = self.enc1_2(enc1_1)  # connect dec1_1
        enc_pool1 = self.enc_pool1(enc1_2)
        
        enc2_1    = self.enc2_1(enc_pool1)
        enc2_2    = self.enc2_2(enc2_1)  # connect dec2_1
        enc_pool2 = self.enc_pool1(enc2_2)

        enc3_1    = self.enc3_1(enc_pool2)
        enc3_2    = self.enc3_2(enc3_1)  # connect dec3_1
        enc_pool3 = self.enc_pool1(enc3_2)

        enc4_1    = self.enc4_1(enc_pool3)
        enc4_2    = self.enc4_2(enc4_1)  # connect dec4_1
        enc_pool4 = self.enc_pool1(enc4_2)

        enc5_1    = self.enc5_1(enc_pool4)

        dec5_1      = self.dec5_1(enc5_1)

        dec_unpool4 = self.dec_unpool4(dec5_1)
        dec4_1      = self.dec4_1( torch.cat( ( dec_unpool4 , enc4_2) , dim = 1 ) ) 
        dec4_2      = self.dec4_2(dec4_1)
        
        dec_unpool3 = self.dec_unpool3(dec4_2)
        dec3_1      = self.dec3_1( torch.cat( ( dec_unpool3, enc3_2) , dim = 1 ) ) 
        dec3_2      = self.dec3_2(dec3_1)

        dec_unpool2 = self.dec_unpool2(dec3_2)
        dec2_1      = self.dec2_1( torch.cat( ( dec_unpool2, enc2_2) , dim = 1 ) ) 
        dec2_2      = self.dec2_2(dec2_1)

        dec_unpool1 = self.dec_unpool1(dec2_2)
        dec1_1      = self.dec1_1( torch.cat( ( dec_unpool1, enc1_2) , dim = 1 ) ) 
        dec1_2      = self.dec1_2(dec1_1)

        output = self.last(dec1_2)

        return output 