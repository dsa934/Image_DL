
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-14

 < Facades dataset >

  - image size : (W,H) = (512, 256)

                 (256,256) image x 2 

                 (0 ~ 255, 256)   : photo img

                 (256 ~ 512, 256) : draw img 
                 

  - Data ratio : train / val / test = 400 / 100 / 106

                 validation data �� test data ���� ����ϰ� ����������(100��) 

                 �н� ������ : train + test ,  �׽�Ʈ ������ : val ��� 




 < �˾Ƶθ� ���� method >

  - glob.glob()

    => Ư���� ���� or Ȯ���ڸ� ���� ���ϵ��� ��� or �̸��� �ʿ��� ��� ���

       �ۼ��ڰ� ������ ���ǿ� �´� ���ϸ� or ��θ� list ���·� ��ȯ 

       ���Խ� ��� x , '*' , '?' ���� ���� ( �ᱹ �ݺ��� ���ϱ� ���� ��� �ϴ� library)


  - PIL's Image crop()

    => crop (  (���ϴ� ��ǥ , ���� ��ǥ )   )
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

        # train + test data => �н� ������ ( �������� ���� ���� ����)
        if mode == 'train' : 

            self.img_files.extend ( sorted( glob.glob( os.path.join(data_path, 'test') + "/*.jpg")) )
    

    def __len__(self):
        return len(self.img_files)


    def __getitem__(self, index):
        
        img = Image.open(self.img_files[ index % len(self.img_files)])

        w, h = img.size

        photo_img = img.crop( (0, 0, w/2, h) )
        draw_img  = img.crop( (w/2, 0, w, h) )

        # �¿� ����(Horizontal flip) for Data augmentation
        if np.random.random() < 0.5 : 

            photo_img = Image.fromarray( np.array(photo_img)[:, ::-1, :] , 'RGB')
            draw_img  = Image.fromarray( np.array(draw_img)[ :, ::-1, :] , 'RGB')

        photo_img, draw_img = self.transforms(photo_img), self.transforms(draw_img)

        _dataset ={'photo' : photo_img, 'draw' : draw_img}

        return _dataset


# image �� pair�� ���·� ������ �ʰ�, [256,256] 1�常 ������ ��� ���
# pair�� ������ case�� �� Customdatasets �̿� 
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
sample image ��� 

image = Image.open('./Pix2Pix/data/train/1.jpg')
print(f"image shape : {image.size}")
plt.imshow(image)
plt.show()
'''