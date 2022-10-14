
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-14

 < Facades dataset >

  - image size : (W,H) = (512, 256)

                 (256,256) image x 2 

                 (0 ~ 255, 256)   : photo img

                 (256 ~ 512, 256) : draw img 
                 

  - Data ratio : train / val / test = 400 / 100 / 106

                 validation data 가 test data 보다 깔끔하게 떨어짐으로(100개) 

                 학습 데이터 : train + test ,  테스트 데이터 : val 사용 




 < 알아두면 좋은 method >

  - glob.glob()

    => 특정한 패턴 or 확장자를 가진 파일들의 경로 or 이름이 필요할 경우 사용

       작성자가 제시한 조건에 맞는 파일명 or 경로를 list 형태로 반환 

       정규식 사용 x , '*' , '?' 지원 가능 ( 결국 반복을 피하기 위해 사용 하는 library)


  - PIL's Image crop()

    => crop (  (좌하단 좌표 , 우상단 좌표 )   )
'''

import os
import glob
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset


class CustomDatasets(Dataset):

    def __init__(self, data_path, transforms = None, mode = 'train' ):

        self.transforms = transforms
        
        self.img_files = sorted( glob.glob( os.path.join(data_path, mode) + "/*.jpg" ) ) 

        # train + test data => 학습 데이터 ( 데이터의 수가 적기 때문)
        if mode == 'train' : 

            self.img_files.extend ( sorted( glob.glob( os.path.join(data_path, 'test') + "/*.jpg")) )
    

    def __len__(self):
        return len(self.img_files)


    def __getitem__(self, index):
        
        img = Image.open(self.img_files[ index % len(self.img_files)])

        w, h = img.size

        photo_img = img.crop( (0, 0, w/2, h) )
        draw_img  = img.crop( (w/2, 0, w, h) )

        # 좌우 반전(Horizontal flip) for Data augmentation
        if np.random.random() < 0.5 : 

            photo_img = Image.fromarray( np.array(photo_img)[:, ::-1, :] , 'RGB')
            draw_img  = Image.fromarray( np.array(draw_img)[ :, ::-1, :] , 'RGB')

        photo_img, draw_img = self.transforms(photo_img), self.transforms(draw_img)

        _dataset ={'photo' : photo_img, 'draw' : draw_img}

        return _dataset


# image 가 pair의 형태로 들어오지 않고, [256,256] 1장만 들어오는 경우 사용
# pair로 들어오는 case는 위 Customdatasets 이용 
class CustomDatasetsMyData(Dataset):

    def __init__(self, data_path, transforms = None, mode = None):

        self.transforms = transforms
        
        self.img_files = sorted( glob.glob( data_path + "/*.png" ) ) 


    def __len__(self):
        return len(self.img_files)


    def __getitem__(self, index):

        img = Image.open(self.img_files[ index % len(self.img_files)])

        img = self.transforms(img)

        _dataset ={'custom' : img}

        return _dataset




'''
sample image 출력 

image = Image.open('./Pix2Pix/data/train/1.jpg')
print(f"image shape : {image.size}")
plt.imshow(image)
plt.show()
'''