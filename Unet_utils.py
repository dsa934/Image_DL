# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-06


 < Utils >

  - torch�� built-in function�� ���� �����غ��� ( ToTensor, Normalize, �� )

  - https://www.youtube.com/watch?v=1gMnChpUS9k&list=PLqtXapA2WDqbE6ghoiEJIrmEnndQ7ouys&index=5 (����)


 < JW Dataset >

  - torchvision �� datasets �� �̹� ����� �����͸� ������ ����ϴ� ���

    => �ش� ������ ���, ������ �����͸� Ȱ���ϱ� ������ Custom dataset class �ʿ� 



 < ���� �� ���� ���� >

  - �Ϲ����� ������ �ƴ� ��� ( ex, file, JSON, ... )

     => �Ժη� ���� �Լ���� ������ �Ű������� ���ļ� ����� ��� None�� ������ �� �������� ���� �и��ؼ� ��� 

    
'''

import os

from matplotlib.colors import Normalize
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'



# ToTensor ���� ( Numpy -> Tensor )
class JW_ToTensor(object):

    def __call__(self, data):

        input_data, label = data['data'] , data['label']

        # numpy shape = [H,W,C] , tensor shape = [ C, H, W]
        input_data = input_data.transpose((2,0,1)).astype(np.float32)
        label = label.transpose((2,0,1)).astype(np.float32)

        data = {'data' : torch.from_numpy(input_data) , 'label' : torch.from_numpy(label)}

        return data


class JW_Normalize(object):

    def __init__(self, mean = 0.5, std = 0.5):

        self.mean = mean
        self.std = std

    def __call__(self,data):

        input_data, label = data['data'] , data['label']

        input_data = (input_data - self.mean) / self.std

        data = {'data' : input_data , 'label' : label }

        return data


class JW_RandomFlip(object):

    def __call__(self,data):

        input_data, label = data['data'] , data['label']

        # fliplr() : numpy �¿� ���� 
        if np.random.rand() > 0.5 :

            input_data = np.fliplr(input_data)
            label = np.fliplr(label)

        # flipud() : numpy ���� ����
        if np.random.rand() > 0.5 :
            input_data = np.flipud(input_data)
            label = np.flipud(label)


        output_data = {'data' : input_data , 'label' : label }

        return output_data

    

class JW_Dataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form 
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir, data_type='float32', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_type = data_type

        lst_data = os.listdir(data_dir)

        lst_input = [f for f in lst_data if f.startswith('data')]
        lst_label = [f for f in lst_data if f.startswith('label')]

        lst_input.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        lst_label.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.lst_input = lst_input
        self.lst_label = lst_label

    def __getitem__(self, index):

        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        _input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        if label.dtype == np.uint8:
            label = label / 255.0

        if _input.dtype == np.uint8:
            _input = _input / 255.0

        if label.ndim == 2:
            label = np.expand_dims(label, axis=2)

        if _input.ndim == 2:
            _input = np.expand_dims(_input, axis=2)

        input_data = {'data': _input, 'label': label}

        if self.transform:
            
            input_data = self.transform(input_data)
            
        return input_data

    def __len__(self):
        return len(self.lst_label)


# ������ �δ� ���� �����غ��� 
class JW_Dataloader(torch.utils.data.Dataset):

    def __init__(self, data_path, transform = None):

        self.data_path = data_path
        self.transform = transform

        
        # ������ ���� ��ο� �ִ� �����͸� list�� �ҷ���
        list_data = os.listdir(self.data_path)

        # train dir�� train_data, train_label �� ���� �и�
        list_train = [ f for f in list_data if f.startswith('data_')]
        list_label = [ f for f in list_data if f.startswith('label_')]

        list_train.sort()
        list_label.sort()


        self.list_label = list_label
        self.list_train = list_train


    def __len__(self):
        return len(self.list_label)


    def __get__(self, index):

        _data = np.load(os.path.join(self.data_path, self.list_train[index]))
        _label = np.load(os.path.join(self.data_path, self.list_label[index]))

        # Normalize (���� �����Ͱ� 0 ~ 255 ������ 0~1 ���� �ٻ�)
        _data, _label = _data/255. , _label/255.

        # NN �Է� �������� ���� = [height, weight, channel], ä���� ���� ��� �߰��Ͽ� 3D tensor ���·� �����ؾ� �� 
        if _data.ndim == 2 : _data = _data[:, :, np.newaxis ]
        if _label.ndim == 2 : _label = _label[:, :, np.newaxis ]

        input_data = {'data' : _data, 'label' : _label }

        if self.transform: input_data = self.transform(input_data)


        return input_data

 
    
## ���� ������ util function ���� �� �۵��ϴ��� Ȯ���ϱ� ���� ���� 

#from torchvision import transforms

#transform = transforms.Compose( [ JW_Normalize(mean=0.5, std=0.5), JW_RandomFlip(), JW_ToTensor() ] ) 

#JW_Collect_Data()


# Dataloader ������ ���������� üũ  =>  data image���� Yellow(1) , Dark(0) 
#train_data = JW_Dataloader(data_path = './Unet_data/train', transform = transform)
#data = train_data.__get__(0)


#train, label = data['data'] , data['label']

# ToTensor  ������ ���������� ��ٸ� ����� = [1,512,512]
#print(train.shape)

#plt.subplot(121)
#plt.imshow(train.squeeze() )
#plt.title("data")

#plt.subplot(122)
#plt.imshow(label.squeeze())
#plt.title("label")

#plt.show()

