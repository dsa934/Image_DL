# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-06


< Collect Data >

 1. ISBI CHallenge : Segmentation of neuronal structures in EM stacks  ������ �ٿ�ε�

    => �ش� ������ ���� membrane(������) ���θ� �Ǵ� ( 0 : ������ x , 255 : ������ o )
 
 2. train-labels.tif, train-volume.tif , ���Ϸ� ����   train : val : test = 24 : 3 : 3 ���� ��� 

 3. train , val, test dir�� ������ ���� 
   

 < function for collect data  >

  - PIL's Image function 

    * seek 

      => img.seek(idx = m) , m ����  frame �̹����� �ҷ�����  

         ��  np.asarray( img.seek(idx) ) �� ���� �Ű������� ����� ��� ,

         None ���� ��� �Ǳ� ������  seek�� �Ű������� �־ ������� ���� 


    * Image

      => Raw data dim = [ 512, 512, 30 ]  :  512 x 512 gray-scale image, 30 frames

         img_train & img_ label = [512, 512]

    * .n_frames 

      => Image�� open�� �̹��� �������� ������ ���� 


    * .seek(idx) 

      => idx��(frame ����) �ش��ϴ� image ã�� 


  - Mmatplotlib.pyplot 

    => ���̺귯�� �浿 ���� �߻� : OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. 

       �ذ� : os.environ['KMP_DUPLICATE_LIB_OK']='True' 

'''

import os

from matplotlib.colors import Normalize
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'



def JW_Collect_Data():

    # Load Data
    # os.path.join(directory, file_name) -> file name�� ���� �Է½� \\ �ߺ� �߻� 
    # data dimension = [ 512, 512, 30 ] 30 frame, gray-scale
    dir_, name_train, name_label  = './Unet' , 'train-volume.tif' , 'train-labels.tif'
    img_train = Image.open(os.path.join(dir_, name_train))
    img_label = Image.open(os.path.join(dir_, name_label))

    # Save Data
    # train, val, test data ������ ��� ���� 
    dir_train = os.path.join(dir_, 'train')
    dir_val   = os.path.join(dir_, 'val')
    dir_test  = os.path.join(dir_, 'test')

    # dir ����
    if not os.path.exists(dir_train) : os.makedirs(dir_train) 
    if not os.path.exists(dir_val) : os.makedirs(dir_val) 
    if not os.path.exists(dir_test) :os.makedirs(dir_test) 

    # shuffle ��� ����
    shuffle_idx = np.arange(img_train.n_frames)
    np.random.shuffle(shuffle_idx)

    # train : val : test = 24 : 3 : 3
    for idx, value in enumerate(shuffle_idx):

        # train 
        if idx < 24 : 

            # seek�� �Ű������� �־ ������� �� ��
            img_train.seek(value)
            img_label.seek(value)

            train_data = np.asarray( img_train ) 
            train_label = np.asarray( img_label )

            np.save(os.path.join(dir_train, 'data_%02d.npy' % value), train_data )
            np.save(os.path.join(dir_train, 'label_%02d.npy' % value), train_label )
        
        # val
        elif idx < 27 :

            img_train.seek(value)
            img_label.seek(value)

            val_data = np.asarray( img_train ) 
            val_label = np.asarray( img_label )

            np.save(os.path.join(dir_val, 'data_%02d.npy' % value), val_data )
            np.save(os.path.join(dir_val, 'label_%02d.npy' % value), val_label )

        # test
        else :

            img_train.seek(value)
            img_label.seek(value)

            test_data = np.asarray( img_train ) 
            test_label = np.asarray( img_label )

            np.save(os.path.join(dir_test, 'data_%02d.npy' % value), test_data )
            np.save(os.path.join(dir_test, 'label_%02d.npy' % value ), test_label )


    ## chk code for collect data 

    #plt.subplot(121)
    #plt.imshow(train_label, cmap='gray')
    #plt.title('label')

    #plt.subplot(122)
    #plt.imshow(train_data, cmap='gray')
    #plt.title('input')

    #plt.show()       
