# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-06


<  U-Net 복원  >

 - U-Net: Convolutional Networks for Biomedical Image Segmentation (MICCAI 2015)

 - FCN + Skip Connections + Mirroring (미구현)


 < Semantic Segmantation >

  - pixel 단위로 어떤 class에 해당하는지 판별, 해당 예제에서는 세포벽(membrane)인지 아닌지에 대한 판별임으로 Binary Classification에 해당 


 < Fully Convolutional Network (FCN) >

  - 일반적인 classification problem을 위한 NN 구조의 Fully Connected layer를 convolution layer로 대체 

    따라서 FCN은 다음과 같은 특징을 갖게 됨

     a) 입력 이미지의 해상도 제한 x 

     b) 출력 이미지의 해상도는 입력 이미지의 해상도 보다 작다

    위와 같은 FCN의 특징 떄문에 U-net에서는 Overlap-tile 전략을 활용 ()


 < U-Net original paper 와 구현 시 차이점  >

  - Skip Connections   : 원 논문에서는 Contracting path의 feature map을 일부 잘라서 사용하지만, 자르지 않고 사용 

  - Conv layer padding : 원 논문에서는 padding = 0 으로 설정하여, conv_layer 진행마다 이미지 크기가 2씩 감소하지만, 해당 코드에서는 padding = 1로 설정하여 크기가 변하지 않음

                         크기가 변할 경우 skip connection을 위한 cropping 함수가 별도로 필요 함 

  - Last Conv layer    : 원 논문에서는 kernel 수가 2로 끝나지만, 해당 코드에서는 1로 끝나도록 설정 
  
                         데이터 자체가 membrane 인지 아닌지의 여부를 판단하는 Binary classification problem 임으로, 1로 설정하고 BCE Loss를 사용 

  - BCEWithLogitsLoss  : BCEloss를 사용할 경우, 해당 코드의 NN 구성은 마지막 layer에서 activation fucntion을 통과하지 않기 때문에, 

                         loss를 계산하기 위해 필요한 label, data 간 data type이 다를 수 있음 ( Long , float) 이를 보정하기 위하여 sigmoid + bce loss 인 BCEwithLogitLoss 사용 


  - One obejct         : 해당 코드는 구별해야하는 obejct가 1개임으로,  작은 분리 경계 학습을 위한 W(x) 함수 생략 
    
'''

import os
import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as trans

import matplotlib.pyplot as plt
import numpy as np 

from Unet_data_read import JW_Collect_Data
from Unet_utils import JW_Dataloader, JW_Normalize, JW_RandomFlip , JW_Dataset, JW_ToTensor
from Unet_model import Unet

## set NN hyper params
lr = 1e-5
n_epoch = 100
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 2
# Unet.model.py 참조 
num_class = 1

## set preprocessing params
data_dir = './Unet_data'

## image plot 을 위한 함수들
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## 데이터 수집 
# 경로 존재 여부 판단 미설정 시, 실행마다 데이터 파일을 형성함으로 이를 방지 
if not os.path.exists( os.path.join(data_dir, 'train')) : JW_Collect_Data()


## 데이터 불러오기 
'''
사용자 정의 Datasets으로 데이터를 구성했기 떄문에 return data type = dict() 

torchvision library function은 dict type 자료형태를 다루지 않기 떄문에 

Compose() 안에 구성되는 모든 데이터 전처리 함수를 자체적으로 구현할 필요성 존재 (Unet_utils.py)

순서 중요 : Norm -> RandomFlip -> ToTensor
'''
train_form = trans.Compose( [JW_Normalize(mean = 0.5 ,std =0.5) ,  JW_RandomFlip(), JW_ToTensor() ]  )
train_data = JW_Dataset(data_dir = os.path.join(data_dir, 'train') , transform = train_form)
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = False)

val_data = JW_Dataset(data_dir = os.path.join(data_dir, 'val') , transform = train_form)
val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False )


test_form = trans.Compose( [JW_Normalize(mean = 0.5, std = 0.5 ) , JW_ToTensor() ] )
test_data = JW_Dataset(data_dir = os.path.join(data_dir, 'test'), transform = test_form)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)


## set model instance
model = Unet(num_class).to(device)

## loss & optimizer 설정
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters() , lr = lr)


def train(epoch):

    model.train()

    train_loss = 0

    print(f"Train Epoch : {epoch}\n")
    for batch_idx, train_value in enumerate(train_loader):

        train_data, train_label = train_value['data'].to(device) , train_value['label'].to(device)

        # init optimizer 
        optimizer.zero_grad()

        # cal models's output
        output = model(train_data)

        # train loss
        loss = criterion(output, train_label)
        loss.backward()
        optimizer.step()

        # cal train loss
        train_loss += loss.item()

        print(f"Batch : {batch_idx} | Loss : {loss.item()}" )

    print(f" Avg Train Loss : {train_loss / len(train_loader) }") 


def validation(max_val_loss):

    # validation
    with torch.no_grad():

        model.eval()

        total_val_loss = 0

        for batch_idx, val_value in enumerate(val_loader):

            val_data, val_label = val_value['data'].to(device) , val_value['label'].to(device)

            val_output = model(val_data)

            loss = criterion(val_output, val_label)
            
            total_val_loss += loss.item()

            print(f"Batch : {batch_idx} | Val Loss : {loss.item()}" )

        avg_val_loss = total_val_loss / len(val_loader)

        print(f" Avg Val Loss : { avg_val_loss }") 

    # model save
    if avg_val_loss < max_val_loss : 

        print("New model save !\n")

        max_val_loss = avg_val_loss

        torch.save(model.state_dict(), "./Unet_data/Unet.pt")


    return max_val_loss


def test(model):

    # load model
    model.load_state_dict(torch.load("./Unet_data/Unet.pt"))

    with torch.no_grad():

        model.eval()

        test_loss = 0 

        for batch_idx, test_value in enumerate(test_loader):

            test_data, test_label = test_value['data'].to(device), test_value['label'].to(device)

            output = model(test_data)

            loss = criterion(output, test_label)

            test_loss += loss.item()

            print(f"Batch : {batch_idx} | Test Loss : {loss.item()}" )

            # 결과 이미지 저장
            save_dir = './Unet_data/result'

            test_data = fn_tonumpy(fn_denorm(test_data, mean=0.5, std=0.5))
            test_label = fn_tonumpy(test_label)
            output = fn_tonumpy(fn_class(output))

            for idx in range(test_label.shape[0]):

                _id = len(test_loader) * (batch_idx) + idx 

                # png 파일 저장
                plt.imsave(os.path.join(save_dir, 'png', 'test_data_%03d.png' % _id ) ,test_data[idx].squeeze(), cmap = 'gray' )
                plt.imsave(os.path.join(save_dir, 'png', 'test_label_%03d.png' % _id ) ,test_label[idx].squeeze(), cmap = 'gray' )
                plt.imsave(os.path.join(save_dir, 'png', 'test_output_%03d.png' % _id ) ,output[idx].squeeze(), cmap = 'gray' )

                # numpy 파일 저장 
                np.save(os.path.join(save_dir, 'numpy', 'test_data_%03d.npy' % _id ) ,test_data[idx].squeeze())
                np.save(os.path.join(save_dir, 'numpy', 'test_label_%03d.npy' % _id ) ,test_label[idx].squeeze())
                np.save(os.path.join(save_dir, 'numpy', 'test_output_%03d.npy' % _id ) ,output[idx].squeeze())


## training
# best model setup을 위한 val_loss
max_val_loss = float('inf')

for epoch in range(n_epoch):

    train(epoch)
    max_val_loss = validation(max_val_loss)

 
## test
test(model)

