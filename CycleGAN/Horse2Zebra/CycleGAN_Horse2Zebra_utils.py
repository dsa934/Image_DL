
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-19


< 논문에서 서술된 Network 구조  >
 We apply two techniques from recent works to stabilize our model training procedure. 
 First, for LGAN (Equation 1), we replace the negative log likelihood objective by a least-squares loss [35]. This loss is more stable during training and generates higher quality results.
 Second, to reduce model oscillation [15], we follow Shrivastava et al.’s strategy [46] and update the discriminators using a history of generated images rather than the ones produced by the latest generators.
 We keep an image buffer that stores the 50 previously created images.

* img size = [256, 256]
* train_horse / train_zebra : 1067 / 1334
* test horse / test zebra   : 120 / 140 
* weight are initialized from a Gaussian Distribution N(0 ,0.02)

'''
import os
import glob
import numpy as np 
from PIL import Image 
from torch.utils.data import Dataset

import torch


## BW image - > Color image
# dataset color part에 BW img가 포함되어 있음 
def BW2RGB(image):

    rgb = Image.new('RGB', image.size)
    rgb.paste(image)

    return rgb


class ImageDataset(Dataset):

    def __init__(self, data_path, transform , mode):

        self.transform = transform

        self.horsefile = sorted( glob.glob(os.path.join(data_path, f'{mode}_horse') + '/*.jpg' ) )
        self.zebrafile = sorted( glob.glob(os.path.join(data_path, f'{mode}_zebra') + '/*.jpg'  ) )


    def __getitem__(self, index):

        img_horse = Image.open( self.horsefile[index % len(self.horsefile) ])
        img_zebra  = Image.open( self.zebrafile[ np.random.randint(0, len(self.zebrafile)-1) ])

        if img_horse.mode != 'RGB' : 

            img_horse = BW2RGB(img_horse)

        if img_zebra.mode != 'RGB' : 

            img_zebra = BW2RGB(img_zebra)


        img_horse = self.transform(img_horse)
        img_zebra = self.transform(img_zebra)

        data = {'horse' : img_horse, 'zebra' : img_zebra}

        return data

    # dataloader 함수에서 필요 
    def __len__(self):

        return max( len(self.horsefile) , len(self.zebrafile) )


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