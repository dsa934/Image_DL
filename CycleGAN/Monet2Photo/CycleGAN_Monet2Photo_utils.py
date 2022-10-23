
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-21

* img size = [256, 256]
* train_monet / train_photo :  6287 / 6287
* test horse / test zebra   : 121 / 121
* weight are initialized from a Gaussian Distribution N(0 ,0.02)

'''
import os
import glob
import numpy as np 
from PIL import Image 
from torch.utils.data import Dataset

import torch


## BW image - > photo image
# dataset photo part에 BW img가 포함되어 있음 
def BW2RGB(image):

    rgb = Image.new('RGB', image.size)
    rgb.paste(image)

    return rgb


class ImageDataset(Dataset):

    def __init__(self, data_path, transform , mode):

        self.transform = transform

        self.photofile = sorted( glob.glob(os.path.join(data_path, f'{mode}_photo') + '/*.jpg' ) )
        self.monetfile = sorted( glob.glob(os.path.join(data_path, f'{mode}_monet') + '/*.jpg'  ) )


    def __getitem__(self, index):

        img_photo = Image.open( self.photofile[index % len(self.photofile) ])
        img_monet  = Image.open( self.monetfile[ np.random.randint(0, len(self.monetfile)-1) ])

        if img_photo.mode != 'RGB' : 

            img_photo = BW2RGB(img_photo)

        if img_monet.mode != 'RGB' : 

            img_monet = BW2RGB(img_monet)


        img_photo = self.transform(img_photo)
        img_monet = self.transform(img_monet)

        data = {'photo' : img_photo, 'monet' : img_monet}

        return data

    # dataloader 함수에서 필요 
    def __len__(self):

        return max( len(self.photofile) , len(self.monetfile) ) 


class Buffer():

    def __init__(self, max_size = 50):
        
        self.max_size = max_size
        self.field = [] 
        self.return_img = None
        
    def push_pop(self, data):
        
        # init data dim = [6, 3, 128 , 128]

        target_batch = data.shape[0] 

        if len(self.field) < self.max_size :

            for img in data:

                self.field.append(img)

            return data

        else:

            for idx in range(target_batch):

                index = np.random.randint(0, self.max_size-1)


                if idx == 0 : 

                    self.return_img = self.field[index].unsqueeze(0)

                else:

                    self.return_img = torch.cat( (self.return_img, self.field[index].unsqueeze(0) ) , dim = 0 ) 

            return self.return_img



## 시간에 흐름에 따른 lr 감소 클래스 
class LambdaLR:
    def __init__(self, n_epochs, decay_start_epoch):

        self.n_epochs = n_epochs

        self.decay_start_epoch = decay_start_epoch 

    def step(self, epoch):
        return 1.0 - max(0, epoch - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


## model 가중치 초기화 
def weights_init_normal(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:

        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

        if hasattr(m, "bias") and m.bias is not None:

            torch.nn.init.constant_(m.bias.data, 0.0)

    elif classname.find("BatchNorm2d") != -1:

        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)

        torch.nn.init.constant_(m.bias.data, 0.0)