
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-14


< Pix2Pix 복원  >

 - Image-to-Image Translation with Conditional Adversarial Networks (CVPR 2017)


< Characteristics > 

 - Generator     : U-net Structure

 - Discriminator : Convolutional Patch GAN

 - Generator Loss = cGAN Loss + pixel_Lambda * L1 Loss

   * cGAN Loss : 현실적인 이미지를 만들도록 유도  ( generated_img + conditional_img & gt_real )

   * L1 Loss   : 실제 정답과 유사하도록 유도  ( generated_img  & photo_img )


 - display image
 
   =>  raw data dim   : (W, H, C) = (512, 256, 3) 

       after ToTensor : (C, H, W) = (3, 256, 512)

       따라서 torch.cat( (condition_img, generated_img, photo_img ) , dim = - 2 ) 는 height 를 기준으로 이미지를 이어 붙이는 것을 의미 


 - 한계점 

   => paired data : 서로 다른 두 도메인 x,y의 데이터가 한쌍으로 존재해야 학습이 가능 


< Method >

 - trans.InterpolationMode.BICUBIC  ( Image.BICUBIC is past ver.)

   => 보간법(Interpolation)의 일종 

      이미지의 기하학적 변환(확대, 회전 등) 시, 원본 영상의 정보를 받지 못하는 pixel 발생 (hole)

      hole 주변의 알고 있는 값을 이용하여 hole 값을 유도 


'''
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as trans
from torchvision.utils import save_image

from Pix2Pix_utils import CustomDatasets, CustomDatasetsMyData
from Pix2Pix_model import Generator, Discriminator


## set NN's hyper params
lr = 3e-5
batch_size = 10
n_epoch = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pixel_lambda = 100

## set preprocessing params
data_path = './Pix2Pix/data'
save_path = './Pix2Pix/result'

## set data format 
data_transform = trans.Compose([
    # Image.BICUBIC := image interpolation 
    trans.Resize( (256, 256) , trans.InterpolationMode.BICUBIC ),
    trans.ToTensor(),
    trans.Normalize( (0.5, 0.5, 0.5) , (0.5, 0.5, 0.5) ) 
    ])

## load data
train_data = CustomDatasets(data_path, data_transform)
val_data   = CustomDatasets(data_path, data_transform, 'val')

## train_loader = 400 + 106 / 10 = 51 , val_loader = 100 / 10 = 10 
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True )
val_loader  = DataLoader(val_data, batch_size = batch_size, shuffle = True )


## model set up
generator = Generator().to(device)
discriminator = Discriminator().to(device)

## loss set up
criterion = nn.MSELoss().to(device)
pixel_criterion = nn.L1Loss().to(device)

## optimizer set up
generator_optim = optim.Adam(generator.parameters(), lr = lr, betas = (0.5, 0.999)  )
discriminator_optim = optim.Adam(discriminator.parameters(), lr = lr, betas = (0.5 , 0.999) )


## set train func
def train(epoch):

    for batch_idx, data in enumerate(train_loader):

        # both img dim = [10, 3, 256, 256]
        photo_img , condition_img = data['photo'].to(device) , data['draw'].to(device)

        # ground truth
        # discriminator's output dim = [ batch, 1, 16, 16]
        gt_real = torch.cuda.FloatTensor( photo_img.shape[0] ,1 , 16 ,16 ).fill_(1.0)
        gt_fake = torch.cuda.FloatTensor( condition_img.shape[0], 1, 16, 16).fill_(0.0)

        # Generator training
        generator_optim.zero_grad()

        # condition img := 손 그림  , 손그림을 사진 처럼 바꿔주는 Task 임으로 손그림이 input data
        # generated_img dim = [10, 3, 256, 256]
        generated_img = generator(condition_img)

        # Generator loss = cGAN loss +  pixel_labmda * L1 loss
        cGAN_loss = criterion( discriminator( generated_img, condition_img ) , gt_real ) 
        L1_loss = pixel_criterion(generated_img, photo_img)
        g_loss = cGAN_loss + pixel_lambda * L1_loss

        g_loss.backward()
        generator_optim.step()


        # Discriminator training
        discriminator_optim.zero_grad()

        real_loss = criterion ( discriminator(photo_img, condition_img),  gt_real) 
        fake_loss = criterion ( discriminator(generated_img.detach() , condition_img) , gt_fake)

        d_loss = ( real_loss + fake_loss) /2 
        d_loss.backward()
        discriminator_optim.step()

        
        if epoch % 20 == 0 and batch_idx % 25 == 0  :

            # condition image  |  model output | label image
            # display_image dim = [10, 3, 256 * 3 , 256  ]
            display_image = torch.cat ( (condition_img, generated_img, photo_img) , dim = -2 )
            save_image(display_image[:5], f'./Pix2Pix/result/train/{epoch}_{batch_idx}.png', nrow=5, normalize = True)

    print(f'Epoch : {epoch}/{n_epoch}  |  G loss : {g_loss.item()}  |  cGAN loss : {cGAN_loss.item()} |  L1 loss : {L1_loss.item()}  |  D loss : {d_loss.item()}')


    # model save
    if epoch == n_epoch - 1 :

        torch.save ( generator.state_dict() , os.path.join(save_path, 'Pix2Pix_generator.pt') )
        torch.save ( discriminator.state_dict(), os.path.join(save_path, 'Pix2Pix_discriminator.pt') )

        print("Model Saved ! ")
                    


## training 
'''
for epoch in range(n_epoch):
    
    train(epoch)

'''


## Image Generate Test
def test_generate():

    # model init
    generator = Generator().to(device)
    
    generator.eval()

    with torch.no_grad():

        generator.load_state_dict( torch.load( os.path.join(save_path, 'Pix2Pix_generator.pt')))

        # batch 만큼의 image 추출 
        test_img = next(iter(val_loader))
        
        photo_img, condition_img = test_img['photo'].to(device) , test_img['draw'].to(device)

        test_generate_img = generator(condition_img)

        test_display_img = torch.cat( (condition_img, test_generate_img, photo_img) , dim = -2 ) 

        save_image(test_display_img[:5] , f'./Pix2Pix/result/test/test_result.png', nrow = 5, normalize = True)


test_generate()




## Custom Image generate
custom_data_path = './Pix2Pix/result/custom_data'

custom_data = CustomDatasetsMyData(custom_data_path, data_transform)
custom_loader = DataLoader(custom_data, batch_size = batch_size, shuffle = True)

def custom_generate():
   
    # model init
    generator = Generator().to(device)
    
    generator.eval()

    with torch.no_grad():

        generator.load_state_dict( torch.load( os.path.join(save_path, 'Pix2Pix_generator.pt')))

        custom_img = next(iter(custom_loader))
        
        input_img = custom_img['custom'].to(device)
        
        custom_output = generator(input_img)

        _path = os.path.join(custom_data_path, 'custom_result')

        save_image(custom_output, f'{_path}.png', nrow = 5, normalize = True)


custom_generate()