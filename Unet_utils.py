# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-06


 < Utils >

  - torch의 built-in function을 직접 구현해보기 ( ToTensor, Normalize, 등 )

  - https://www.youtube.com/watch?v=1gMnChpUS9k&list=PLqtXapA2WDqbE6ghoiEJIrmEnndQ7ouys&index=5 (참조)


 < JW Dataset >

  - torchvision 의 datasets 은 이미 저장된 데이터를 가져와 사용하는 방법

    => 해당 예제의 경우, 소지한 데이터를 활용하기 때문에 Custom dataset class 필요 



 < 구현 시 주의 사항 >

  - 일반적인 변수가 아닌 경우 ( ex, file, JSON, ... )

     => 함부로 관련 함수들과 파일을 매개변수로 합쳐서 사용할 경우 None을 도출할 수 있음으로 따로 분리해서 사용 

    
'''

import os

from matplotlib.colors import Normalize
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'



# ToTensor 구현 ( Numpy -> Tensor )
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

        # fliplr() : numpy 좌우 반전 
        if np.random.rand() > 0.5 :

            input_data = np.fliplr(input_data)
            label = np.fliplr(label)

        # flipud() : numpy 상하 반전
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


# 데이터 로더 직접 구현해보기 
class JW_Dataloader(torch.utils.data.Dataset):

    def __init__(self, data_path, transform = None):

        self.data_path = data_path
        self.transform = transform

        
        # 데이터 저장 경로에 있는 데이터를 list로 불러옴
        list_data = os.listdir(self.data_path)

        # train dir에 train_data, train_label 을 각각 분리
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

        # Normalize (원본 데이터가 0 ~ 255 임으로 0~1 사이 근사)
        _data, _label = _data/255. , _label/255.

        # NN 입력 데이터의 차원 = [height, weight, channel], 채널이 없는 경우 추가하여 3D tensor 형태로 구축해야 함 
        if _data.ndim == 2 : _data = _data[:, :, np.newaxis ]
        if _label.ndim == 2 : _label = _label[:, :, np.newaxis ]

        input_data = {'data' : _data, 'label' : _label }

        if self.transform: input_data = self.transform(input_data)


        return input_data

 
    
## 직접 구현한 util function 들이 잘 작동하는지 확인하기 위한 예제 

#from torchvision import transforms

#transform = transforms.Compose( [ JW_Normalize(mean=0.5, std=0.5), JW_RandomFlip(), JW_ToTensor() ] ) 

#JW_Collect_Data()


# Dataloader 구현이 정상적인지 체크  =>  data image에서 Yellow(1) , Dark(0) 
#train_data = JW_Dataloader(data_path = './Unet_data/train', transform = transform)
#data = train_data.__get__(0)


#train, label = data['data'] , data['label']

# ToTensor  구현이 정상적으로 됬다면 결과값 = [1,512,512]
#print(train.shape)

#plt.subplot(121)
#plt.imshow(train.squeeze() )
#plt.title("data")

#plt.subplot(122)
#plt.imshow(label.squeeze())
#plt.title("label")

#plt.show()

