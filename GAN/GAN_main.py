
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-27


< GAN(Generative Adversarial Networks) 복원 >

 - Generative Adversarial Nets (NIPS 2014)


< Load MNIST data>

 train_data := 일종의 Meta file의 형태로 저장 된 상태 


 < Objective Function >

  * objective function ≠ loss function 

    => 공통점 : 학습의 최적화를 위해 존재하는 함수

    => 차이점 : loss function(최소화 지향) objective function(최소/최대화 지향) 

  * loss function is special case of objective function


 < GAN's Objective Function 해석>

  min max V(D,G) = E_x~p_data(x) [ log D(x) ]  + E_z~p_z(z) [ log(1-D(G(z))) ]
   G   D

   * General 

     => E : 보통 확률모델은 랜덤하기 때문에 기대값으로 처리

        p_data(x)   : 원본 데이터의 분포 

        x~p_data(x) : 원본 데이터에서 한개의 데이터 x를 sampling 

        p_z(z)      : 하나의 Noise를 sampling 할 수 있는 distribution ( Uniform , Gaussian ...)

        z~p_z(z)    : Noise를 뽑을 수 잇는 distribution으로 부터 1개의 noise data z sampling 

   * Generator 

     => 각 class에 대하여 적절한 데이터의 분포를 학습 ( A statistical model of the joint probability distribution )

     => purpose : minimize V(D,G)  =>  minimize  log(1-D(G(z))) ( 제너레이터가 관여할 수 있는 부분은 후반부 뿐 )

        log(1-D(G(z))) 를 최소화 하려면, D(G(z)) 값이 1이 되도록 학습이 되야 한다.

        즉, Generated_img 는 discriminator가 Real_img(1) 라고 인식할 수 있도록 학습을 시킨다.


   * Discriminator

     => decision boundary를 학습하는 것 

     => purpose : maximize E_x~p_data(x) [ log D(x) ]  + E_z~p_z(z) [ log(1-D(G(z))) ]

        목적 함수를 최대화 하기 위해서는, log( D(x) ) 와  log(1-D(G(z))) 를 최대화 해야 한다. ( 목적함수 ≠ 손실함수)

        log( D(x) )    : D(x) 값이 1에 수렴해야 해당식이 최대화 => discriminator 관점에서 원본 데이터 분포로부터 생성된 이미지는 real_img(1)로 학습되어야 함 

        log(1-D(G(z))) : D(G(z)) 값이 0에 수렴해야 해당식이 최대화 => discriminator 관점에서 z로부터 생성된 이미지는 fake_img(0)으로 학습되어야 함 
        

'''

import torch
import torch.nn as nn
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)

from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torchvision.transforms as trans

from GAN_model import Generator, Discriminator


# train data format = [size, tensor, normal(mean, std)] 
train_data_form = trans.Compose( [ trans.Resize(28), trans.ToTensor(), trans.Normalize([0.5], [0.5]) ] )

# MNIST DATASET : 60,000 (train), 10,000(test)
# train_data => 일종의 meta fild 형태로 저장
train_data = MNIST( root = './dataset', train = True, download = True, transform = train_data_form )
# len(data_loader) = 60,000 / 128 = 468.75 
data_loader = torch.utils.data.DataLoader(train_data, batch_size = 128, shuffle = True )

# set hyper params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learing_rate = 0.0005
batch_size = 128 
n_epoch = 200
# 생성자에서 임의의 noise sample을 뽑을 떄 사용할 벡터의 크기 
laten_input_dim = 100

# set NN instance
generator = Generator(laten_input_dim).to(device)
discriminator = Discriminator().to(device)

# set loss
criterion = nn.BCELoss().to(device)

# set optim
ge_optimizer = optim.Adam(generator.parameters(), lr = learing_rate)
dis_optimizer = optim.Adam(discriminator.parameters(), lr = learing_rate) 

# training
for epoch in range(n_epoch):

    for i , (_img, _) in enumerate(data_loader):

        # _img dim = [128,1,28,28]
        _img = _img.to(device)

        # gt : ground_truth , dim = [128, 1]
        gt_real = torch.cuda.FloatTensor(_img.shape[0],1).fill_(1.0)
        gt_fake = torch.cuda.FloatTensor(_img.shape[0],1).fill_(0.0)


        ## train - generator
        ge_optimizer.zero_grad()

        # random noise sampling , z_dim = [128, 100]
        z = torch.normal(mean=0, std=1, size = (_img.shape[0], laten_input_dim)).to(device)
        
        # _generated_img_dim = [128,1,28,28]
        _generated_img = generator(z)
        
        # log(1-D(G(z))) 
        g_loss = criterion(discriminator(_generated_img), gt_real)

        # update
        g_loss.backward()
        ge_optimizer.step()


        ## train - discriminator
        dis_optimizer.zero_grad()

        # E[ log(D(real_x)) ] + E[ log(1-D(G(z))) ]
        real_loss = criterion(discriminator(_img), gt_real)
        # _generated_img.detach() : gradient 전파 되지 않는 tensor 복제
        fake_loss = criterion(discriminator(_generated_img.detach()), gt_fake)
        dis_loss = (real_loss + fake_loss) / 2

        # update
        dis_loss.backward()
        dis_optimizer.step()


        # 매 50 epoch, 100 iterations 마다 GAN에 의해 생성되는 img 저장 
        if epoch % 50 == 0 and i % 100 == 0  :

            # 매 50 epoch , 50 iterator 마다  생성된 이미지 25장을 5 x 5 격자 형태로 출력 
            save_image(_generated_img[:25], f'./original_GAN_image_output/{epoch}_{i}.png', nrow=5, normalize = True)

    print(f'Epoch : {epoch}/{n_epoch}  |  G loss : {g_loss}  | D loss : {dis_loss}')


