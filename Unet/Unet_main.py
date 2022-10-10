# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-06


<  U-Net ����  >

 - U-Net: Convolutional Networks for Biomedical Image Segmentation (MICCAI 2015)

 - FCN + Skip Connections + Mirroring (�̱���)


 < Semantic Segmantation >

  - pixel ������ � class�� �ش��ϴ��� �Ǻ�, �ش� ���������� ������(membrane)���� �ƴ����� ���� �Ǻ������� Binary Classification�� �ش� 


 < Fully Convolutional Network (FCN) >

  - �Ϲ����� classification problem�� ���� NN ������ Fully Connected layer�� convolution layer�� ��ü 

    ���� FCN�� ������ ���� Ư¡�� ���� ��

     a) �Է� �̹����� �ػ� ���� x 

     b) ��� �̹����� �ػ󵵴� �Է� �̹����� �ػ� ���� �۴�

    ���� ���� FCN�� Ư¡ ������ U-net������ Overlap-tile ������ Ȱ�� ()


 < U-Net original paper �� ���� �� ������  >

  - Skip Connections   : �� �������� Contracting path�� feature map�� �Ϻ� �߶� ���������, �ڸ��� �ʰ� ��� 

  - Conv layer padding : �� �������� padding = 0 ���� �����Ͽ�, conv_layer ���ึ�� �̹��� ũ�Ⱑ 2�� ����������, �ش� �ڵ忡���� padding = 1�� �����Ͽ� ũ�Ⱑ ������ ����

                         ũ�Ⱑ ���� ��� skip connection�� ���� cropping �Լ��� ������ �ʿ� �� 

  - Last Conv layer    : �� �������� kernel ���� 2�� ��������, �ش� �ڵ忡���� 1�� �������� ���� 
  
                         ������ ��ü�� membrane ���� �ƴ����� ���θ� �Ǵ��ϴ� Binary classification problem ������, 1�� �����ϰ� BCE Loss�� ��� 

  - BCEWithLogitsLoss  : BCEloss�� ����� ���, �ش� �ڵ��� NN ������ ������ layer���� activation fucntion�� ������� �ʱ� ������, 

                         loss�� ����ϱ� ���� �ʿ��� label, data �� data type�� �ٸ� �� ���� ( Long , float) �̸� �����ϱ� ���Ͽ� sigmoid + bce loss �� BCEwithLogitLoss ��� 


  - One obejct         : �ش� �ڵ�� �����ؾ��ϴ� obejct�� 1��������,  ���� �и� ��� �н��� ���� W(x) �Լ� ���� 
    
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
# Unet.model.py ���� 
num_class = 1

## set preprocessing params
data_dir = './Unet_data'

## image plot �� ���� �Լ���
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## ������ ���� 
# ��� ���� ���� �Ǵ� �̼��� ��, ���ึ�� ������ ������ ���������� �̸� ���� 
if not os.path.exists( os.path.join(data_dir, 'train')) : JW_Collect_Data()


## ������ �ҷ����� 
'''
����� ���� Datasets���� �����͸� �����߱� ������ return data type = dict() 

torchvision library function�� dict type �ڷ����¸� �ٷ��� �ʱ� ������ 

Compose() �ȿ� �����Ǵ� ��� ������ ��ó�� �Լ��� ��ü������ ������ �ʿ伺 ���� (Unet_utils.py)

���� �߿� : Norm -> RandomFlip -> ToTensor
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

## loss & optimizer ����
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

            # ��� �̹��� ����
            save_dir = './Unet_data/result'

            test_data = fn_tonumpy(fn_denorm(test_data, mean=0.5, std=0.5))
            test_label = fn_tonumpy(test_label)
            output = fn_tonumpy(fn_class(output))

            for idx in range(test_label.shape[0]):

                _id = len(test_loader) * (batch_idx) + idx 

                # png ���� ����
                plt.imsave(os.path.join(save_dir, 'png', 'test_data_%03d.png' % _id ) ,test_data[idx].squeeze(), cmap = 'gray' )
                plt.imsave(os.path.join(save_dir, 'png', 'test_label_%03d.png' % _id ) ,test_label[idx].squeeze(), cmap = 'gray' )
                plt.imsave(os.path.join(save_dir, 'png', 'test_output_%03d.png' % _id ) ,output[idx].squeeze(), cmap = 'gray' )

                # numpy ���� ���� 
                np.save(os.path.join(save_dir, 'numpy', 'test_data_%03d.npy' % _id ) ,test_data[idx].squeeze())
                np.save(os.path.join(save_dir, 'numpy', 'test_label_%03d.npy' % _id ) ,test_label[idx].squeeze())
                np.save(os.path.join(save_dir, 'numpy', 'test_output_%03d.npy' % _id ) ,output[idx].squeeze())


## training
# best model setup�� ���� val_loss
max_val_loss = float('inf')

for epoch in range(n_epoch):

    train(epoch)
    max_val_loss = validation(max_val_loss)

 
## test
test(model)

