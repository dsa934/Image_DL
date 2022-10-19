
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-18


< BW2RGB >

 - ��� <-> ���� �̹��� domain ��ȯ�� �н��ϴ� �Ű�������� �ش� �Լ��� train data�� ���, ���� �̹����� ���� ���� ��� 

   CNN ���꿡�� channel ��꿡 ������� ���� �� �ֱ� ������  ��� �̹����� RGB�� ���·θ� ��ȭ��Ű�� '�� ��ȯ'

   BW2RGB :  [1, 150, 150]  ->  [3, 150, 150] ���´� ��ȯ but �̹��� ��ü�� ��� �״�� 

   image.new('mode' , 'dimension', 'color' ) : color ������ �־����� ������ ������ �������� ���� (���)


< ImageDatset >

 - img_gray �� ���Ͽ� index�� �����ϰ� �̴� ���� 

    => CycleGAN�� ������ Unpaired dataset�� ���� Image-to-Image Translation �̴�.

       ���� ./dataset/landscape/train ���� color , gray�� 2���� ������ �����ϸ�, �ش� ������ ������ '������' ��ü�� ���� color, gray image �̴�.

       ���� indexing�� �Ȱ��� �Ѵٸ� ������ ��ü�� ���� ���� <-> ��� ��ȯ������ ������ ����� �߻��Ѵ�.

       �׷��Ƿ�, ���� �ٸ� object�� ���Ͽ� object�� ��ȭ���� �ʰ� style(����, ���)�� �н��Ͽ� �����ϱ� ���ؼ� �������� �̴� ���̴�.


< Buffer >

 - '�����ڰ� ���� ���� 50���� �̹����� �����ϰ�, �̸� �̿��� �Ǻ��ڸ� ������Ʈ' �ܿ� ������ �ڼ��� ����� �Ǿ� ���� �ʱ� ������ ������ �����ؼ� Buffer ���� 

    * Buffer �� �뷮 50���� ��� ä�� ��� : 50���� �̹��� �߿��� �����ϰ� �̾� return 

    * Biffer �� �뷮 50���� ä���� ���� ��� : �� ��쿡 return�� ���� ������, �ش� iteration���� �н��� �Ұ��� ������ �Է��� �״�� return 




< ������ ������ Network ����  >
 We apply two techniques from recent works to stabilize our model training procedure. 
 First, for LGAN (Equation 1), we replace the negative log likelihood objective by a least-squares loss [35]. This loss is more stable during training and generates higher quality results.
 Second, to reduce model oscillation [15], we follow Shrivastava et al.��s strategy [46] and update the discriminators using a history of generated images rather than the ones produced by the latest generators.
 We keep an image buffer that stores the 50 previously created images.

* img size = [150, 150]
* train / color , gray : both 8300 ��
* val /  color, gray : both 100�� 


'''
import os
import glob
import numpy as np 
from PIL import Image 
from torch.utils.data import Dataset

import torch


## BW image - > Color image
# dataset color part�� BW img�� ���ԵǾ� ���� 
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

    # dataloader �Լ����� �ʿ� 
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


