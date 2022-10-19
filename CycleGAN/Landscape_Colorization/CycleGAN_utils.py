
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-18


< BW2RGB >

 - 흑백 <-> 색상 이미지 domain 전환을 학습하는 신경망임으로 해당 함수는 train data에 흑백, 색상 이미지가 섞여 있을 경우 

   CNN 연산에서 channel 계산에 어려움이 있을 수 있기 때문에  흑백 이미지를 RGB의 형태로만 변화시키는 '형 변환'

   BW2RGB :  [1, 150, 150]  ->  [3, 150, 150] 형태는 변환 but 이미지 자체는 흑백 그대로 

   image.new('mode' , 'dimension', 'color' ) : color 정보가 주어지지 않으면 검은색 바탕으로 구성 (흑백)


< ImageDatset >

 - img_gray 에 대하여 index를 랜덤하게 뽑는 이유 

    => CycleGAN의 목적은 Unpaired dataset에 대한 Image-to-Image Translation 이다.

       현재 ./dataset/landscape/train 에는 color , gray의 2개의 파일이 존재하며, 해당 파일의 구성은 '동일한' 물체에 대한 color, gray image 이다.

       따라서 indexing을 똑같이 한다면 동일한 물체에 대한 색상 <-> 흑백 변환임으로 목적에 모순이 발생한다.

       그러므로, 서로 다른 object에 대하여 object는 변화하지 않고 style(색상, 흑백)만 학습하여 적용하기 위해서 랜덤으로 뽑는 것이다.


< Buffer >

 - '생성자가 만든 이전 50개의 이미지를 저장하고, 이를 이용해 판별자를 업데이트' 외에 논문에서 자세히 언급이 되어 있지 않기 떄문에 상상력을 동원해서 Buffer 구현 

    * Buffer 의 용량 50개를 모두 채운 경우 : 50개의 이미지 중에서 랜덤하게 뽑아 return 

    * Biffer 의 용량 50개를 채우지 못한 경우 : 이 경우에 return을 하지 않으면, 해당 iteration에서 학습이 불가능 함으로 입력을 그대로 return 




< 논문에서 서술된 Network 구조  >
 We apply two techniques from recent works to stabilize our model training procedure. 
 First, for LGAN (Equation 1), we replace the negative log likelihood objective by a least-squares loss [35]. This loss is more stable during training and generates higher quality results.
 Second, to reduce model oscillation [15], we follow Shrivastava et al.’s strategy [46] and update the discriminators using a history of generated images rather than the ones produced by the latest generators.
 We keep an image buffer that stores the 50 previously created images.

* img size = [150, 150]
* train / color , gray : both 8300 장
* val /  color, gray : both 100장 


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

        self.colorfile = sorted( glob.glob(os.path.join(data_path, f'{mode}/color') + '/*.jpg' ) )
        self.grayfile = sorted( glob.glob(os.path.join(data_path, f'{mode}/grayscale') + '/*.jpg'  ) )


    def __getitem__(self, index):

        img_color = Image.open( self.colorfile[index % len(self.colorfile) ])
        img_gray  = Image.open( self.grayfile[ np.random.randint(0, len(self.grayfile)-1) ])

        if img_color.mode != 'RGB' : 

            img_color = BW2RGB(img_color)

        if img_gray.mode != 'RGB' : 

            img_gray = BW2RGB(img_gray)


        img_color = self.transform(img_color)
        img_gray = self.transform(img_gray)

        data = {'color' : img_color, 'gray' : img_gray}

        return data

    # dataloader 함수에서 필요 
    def __len__(self):

        return len(self.colorfile)


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


