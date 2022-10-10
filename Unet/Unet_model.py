
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-06

< Data dimension flow >

[batch, 1, 512, 512]      : init data = [batch, channel, width, height]

[batch, 64, 512, 512]     : ä�� ���� ����

[batch, 64, 512, 512]     : ä�� �� ���� 

[batch, 64, 254, 254]     :  2 x 2 max pool

   (����)

[batch, num_class, height, width] :  Semantic segmentation task ������, �ȼ� ���� �з��۾��� �ʿ� �ϸ�,

                                     ������ layer �� ���� ä�� �� = �з� �ؾ��ϴ� Ŭ������ �� 



 *  pool & unpool 

   - ���� ���(Contracting Path) ���� feature map�� ũ�⸦ ���̴� ���� maxpooling(size = n) �� ����

   - Ȯ�� ���(Expanding Path) ���� feature map�� ũ�⸦ �ø��� ���� ConvTranspose2D( ... , stride = n ) �� ����

     ���� ConvTranspose2D���� �ٷ����� ����� ä�� ���� �����ϰ� ���� 



 * nn.Conv2d( ... , padding = 1 ) 

   - ���� �������� padding = 0 ���� ��� ������, Convolutional layer�� ���� �� ������ image_size = [ H-2, W-2] 

     �̹����� ����� �����ϱ� ������ Contracting , Expadning path �� skip connection ���� �� 

     Contracting path���� ������ feature map�� ũ�Ⱑ Expanding path���� �ٷ����� feature map�� ũ�⺸�� Ŀ�� �״�� concatenation�� �Ұ���

     Contracting path�� feature map�� ���� ũ�� �߶� ������ ��� ��

     �׷��� �ش� �ڵ忡���� padding =1 �� �����ν� Contracting path ~ Expanding path ���� �ȼ� ũ���� ��ȭ�� pooling layer ������ �̷���� ( Expanding path ������ concatenation���� �̹��� ũ�� ���� )

     ���� �״�� Contracting path's feature map �� Expanding Path's feature map�� ���� concatenation ���� 

     �� ������ 100% �״�� ������ ��� Cropping �� ���� �κ��� �߰������� ����� �ʿ伺 ���� 


< Connection > 

 - ���� ������ encoder, decoder ������ ���� concatenation 

 - [batch, channel, height, width ] ������ channel �������� concatenation 

   => dec_unpool4 = [batch, 512, height, width ]  

      enc4_2      = [batch, 512,  height, width ]

 - Contracting path 

   * first conv layer  : ä���� �� ����

   * Second conv layer : ä���� �� ����
 
 - Expanding path

   * first conv layer  : ä���� �� ����

     => encoder�� decoder �� ��Į�ڸ��� ������� ���� ������, ä���� ���� �����Ǿ�� ������ 

        Unet�� encoder / decoder layer ������ output�� concatenation �Ǵ� ������ ��ġ�� ������

        decoder�� ù���� conv layer�� concate �� �Է� �����͸� ó�� ������, ä���� ���� 2��� �þ��

        ���� ������ ������ ������, concated data�� ó�������� ä���� ���� ���ҽ�Ű�� ������ ���̴� ���̴�.


   * Second conv layer : ä���� �� ���� 


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
         �������� ������ layer�� ���� �����Ǵ� feature map�� ������ num_class �� ���� (�ش� example ������ 2 )

         �׷��� 2���� ������ ���� �������� 1�� ���� �� 

         1) binary classification problem ������  num_class = 2 ���� + nn.CrossEntropyLoss() �� ����ϴ� �ͺ��� 

            num_class = 1 �� ���� �� , nn.BCEWithLogitsLoss() �� ����ص� ���� ���� ���� 

            * nn.BCEWithLogitsLoss()  = sigmoid activation f + BCE Loss  


         2)  RuntimeError: Expected object of scalar type Long but got scalar type Float for argument #2 'target' in call to _thnn_nll_loss2d_forward

             Model�� output �� activation function�� ��ġ�� �ʱ� ������ label���� data type�� ���� �ʾ� loss ��꿡 ������ �� �� ���� 
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