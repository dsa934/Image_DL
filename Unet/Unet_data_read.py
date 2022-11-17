# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-06


< Collect Data >

 1. ISBI CHallenge : Segmentation of neuronal structures in EM stacks  데이터 다운로드

    => 해당 데이터 셋은 membrane(세포벽) 여부를 판단 ( 0 : 세포벽 x , 255 : 세포벽 o )
 
 2. train-labels.tif, train-volume.tif , 파일로 부터   train : val : test = 24 : 3 : 3 비율 배분 

 3. train , val, test dir에 데이터 저장 
   

 < function for collected data  >

  - PIL's Image function 

    * seek 

      => img.seek(idx = m) , m 번  frame 이미지를 불러오기  

         단  np.asarray( img.seek(idx) ) 와 같이 매개변수로 사용할 경우 ,

         None 값이 출력 되기 때문에  seek를 매개변수에 넣어서 사용하지 말것 


    * Image

      => Raw data dim = [ 512, 512, 30 ]  :  512 x 512 gray-scale image, 30 frames

         img_train & img_ label = [512, 512]

    * .n_frames 

      => Image로 open한 이미지 데이터의 프레임 접근 


    * .seek(idx) 

      => idx에(frame 기준) 해당하는 image 찾기 


  - Mmatplotlib.pyplot 

    => 라이브러리 충동 문제 발생 : OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. 

       해결 : os.environ['KMP_DUPLICATE_LIB_OK']='True' 

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
    # os.path.join(directory, file_name) -> file name에 직접 입력시 \\ 중복 발생 
    # data dimension = [ 512, 512, 30 ] 30 frame, gray-scale
    dir_, name_train, name_label  = './Unet' , 'train-volume.tif' , 'train-labels.tif'
    img_train = Image.open(os.path.join(dir_, name_train))
    img_label = Image.open(os.path.join(dir_, name_label))

    # Save Data
    # train, val, test data 저장할 경로 설정 
    dir_train = os.path.join(dir_, 'train')
    dir_val   = os.path.join(dir_, 'val')
    dir_test  = os.path.join(dir_, 'test')

    # dir 생성
    if not os.path.exists(dir_train) : os.makedirs(dir_train) 
    if not os.path.exists(dir_val) : os.makedirs(dir_val) 
    if not os.path.exists(dir_test) :os.makedirs(dir_test) 

    # shuffle 기능 구현
    shuffle_idx = np.arange(img_train.n_frames)
    np.random.shuffle(shuffle_idx)

    # train : val : test = 24 : 3 : 3
    for idx, value in enumerate(shuffle_idx):

        # train 
        if idx < 24 : 

            # seek를 매개변수에 넣어서 사용하지 말 것
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
