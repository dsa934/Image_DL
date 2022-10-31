
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-31


< Model Structure >

 * Discriminator : PatchGAN

 * Generator 

   - condition vector  (c) 

     [1, 5 ]            :  init c 
       ��
     [1, 5 , 1, 1]      :  change shape 
       ��
     [1, 5, 256, 256 ]  :  repeat(1, 1, x.shape[2], x.shape[3] )

                           5���� data property value�� [256, 256] �̹��� ���·� ä�� ���� 

                           init c = [ 1, 0, 0, 0, 1 ] �̶��, 

                           c[1,0,:,:] =  torch.ones(256,256)

                           c[1,1,:,:] =  torch.zeros(256,256)


   - image data (x)  = [1, 3, 256, 256]

     => �Է� �̹����� channel �� , c vector�� property ���� �������� concatenation 

        2���� �̹����� condition vector�� concatenation �ϱ� ����, ������ �̹��� ũ��� ���� ���� 

        ���� �߿��� ���� # of channel + # of property 






< �˾� �α�>

 * instanceNormd - track_running_stats 

   =>  a boolean value that when set to True, this module tracks the running mean and variance, 
   
       and when set to False, this module does not track such statistics and always uses batch statistics in both training and eval modes


 * �н��� ���� ����ϴ°��̱� ������, �н� ��Ų ���� �������� �����ϰ� ������� �� 

   ��, ���� �����ϴ� ����� �޶� ������, �ش� �� layer�� �����ϴ� �������� ���ƾ� �� 



'''


import torch 
import torch.nn as nn 
import numpy as np 


class ResidualBlock(nn.Module):
    
    def __init__(self, input_dim, output_dim):

        super().__init__()

        self.main = nn.Sequential(

            # pixel ��ȭ x 
            nn.Conv2d(input_dim, output_dim, kernel_size = 3, stride = 1 , padding = 1 , bias = False ),
            nn.InstanceNorm2d(output_dim, affine = True, track_running_stats = True ),
            nn.ReLU(inplace = True),

            nn.Conv2d(output_dim, output_dim, kernel_size = 3, stride = 1 , padding = 1 , bias = False ),
            nn.InstanceNorm2d(output_dim, affine = True, track_running_stats = True ),
            
            
            )

    def forward(self, data):

        return data + self.main(data)

    

class Discriminator(nn.Module):
    
    def __init__(self, img_size = 128, conv_dim = 64, c_dim = 5, repeat_num = 6):

        super().__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(img_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        

    def forward(self, data):

        # data_shape = [] 
        dis_output = self.main(data)

        out_img = self.conv1(dis_output)
        out_cls = self.conv2(dis_output)

        return out_img, out_cls.view(out_cls.shape[0] , out_cls.shape[1]) 


class Generator(nn.Module):
    
    def __init__(self, conv_dim = 64, c_dim = 5, repeat_num = 6 ):

        super().__init__()

        layers = [] 

        layers.append( nn.Conv2d(3+c_dim, conv_dim, kernel_size = 7, stride = 1, padding = 3, bias = False) )
        layers.append( nn.InstanceNorm2d(conv_dim, affine = True, track_running_stats = True) )
        layers.append( nn.ReLU(inplace=True) )
 
        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(curr_dim, curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        
        # x_dim = [1, 3, 256, 256] , c_dim [1,5] , c = [1, 0, 0, 1, 0]

        # c_dim = [1, 5, 1, 1]
        c = c.view(c.size(0), c.size(1), 1, 1)

        # c_dim = [1, 5, 256, 256]
        c = c.repeat(1, 1, x.size(2), x.size(3))
        
        x = torch.cat([x, c], dim=1)

        return self.main(x)
