
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-27


< GAN(Generative Adversarial Networks) ���� >

 - Generative Adversarial Nets (NIPS 2014)


< Load MNIST data>

 train_data := ������ Meta file�� ���·� ���� �� ���� 


 < Objective Function >

  * objective function �� loss function 

    => ������ : �н��� ����ȭ�� ���� �����ϴ� �Լ�

    => ������ : loss function(�ּ�ȭ ����) objective function(�ּ�/�ִ�ȭ ����) 

  * loss function is special case of objective function


 < GAN's Objective Function �ؼ�>

  min max V(D,G) = E_x~p_data(x) [ log D(x) ]  + E_z~p_z(z) [ log(1-D(G(z))) ]
   G   D

   * General 

     => E : ���� Ȯ������ �����ϱ� ������ ��밪���� ó��

        p_data(x)   : ���� �������� ���� 

        x~p_data(x) : ���� �����Ϳ��� �Ѱ��� ������ x�� sampling 

        p_z(z)      : �ϳ��� Noise�� sampling �� �� �ִ� distribution ( Uniform , Gaussian ...)

        z~p_z(z)    : Noise�� ���� �� �մ� distribution���� ���� 1���� noise data z sampling 

   * Generator 

     => �� class�� ���Ͽ� ������ �������� ������ �н� ( A statistical model of the joint probability distribution )

     => purpose : minimize V(D,G)  =>  minimize  log(1-D(G(z))) ( ���ʷ����Ͱ� ������ �� �ִ� �κ��� �Ĺݺ� �� )

        log(1-D(G(z))) �� �ּ�ȭ �Ϸ���, D(G(z)) ���� 1�� �ǵ��� �н��� �Ǿ� �Ѵ�.

        ��, Generated_img �� discriminator�� Real_img(1) ��� �ν��� �� �ֵ��� �н��� ��Ų��.


   * Discriminator

     => decision boundary�� �н��ϴ� �� 

     => purpose : maximize E_x~p_data(x) [ log D(x) ]  + E_z~p_z(z) [ log(1-D(G(z))) ]

        ���� �Լ��� �ִ�ȭ �ϱ� ���ؼ���, log( D(x) ) ��  log(1-D(G(z))) �� �ִ�ȭ �ؾ� �Ѵ�. ( �����Լ� �� �ս��Լ�)

        log( D(x) )    : D(x) ���� 1�� �����ؾ� �ش���� �ִ�ȭ => discriminator �������� ���� ������ �����κ��� ������ �̹����� real_img(1)�� �н��Ǿ�� �� 

        log(1-D(G(z))) : D(G(z)) ���� 0�� �����ؾ� �ش���� �ִ�ȭ => discriminator �������� z�κ��� ������ �̹����� fake_img(0)���� �н��Ǿ�� �� 
        

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
# train_data => ������ meta fild ���·� ����
train_data = MNIST( root = './dataset', train = True, download = True, transform = train_data_form )
# len(data_loader) = 60,000 / 128 = 468.75 
data_loader = torch.utils.data.DataLoader(train_data, batch_size = 128, shuffle = True )

# set hyper params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learing_rate = 0.0005
batch_size = 128 
n_epoch = 200
# �����ڿ��� ������ noise sample�� ���� �� ����� ������ ũ�� 
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
        # _generated_img.detach() : gradient ���� ���� �ʴ� tensor ����
        fake_loss = criterion(discriminator(_generated_img.detach()), gt_fake)
        dis_loss = (real_loss + fake_loss) / 2

        # update
        dis_loss.backward()
        dis_optimizer.step()


        # �� 50 epoch, 100 iterations ���� GAN�� ���� �����Ǵ� img ���� 
        if epoch % 50 == 0 and i % 100 == 0  :

            # �� 50 epoch , 50 iterator ����  ������ �̹��� 25���� 5 x 5 ���� ���·� ��� 
            save_image(_generated_img[:25], f'./original_GAN_image_output/{epoch}_{i}.png', nrow=5, normalize = True)

    print(f'Epoch : {epoch}/{n_epoch}  |  G loss : {g_loss}  | D loss : {dis_loss}')


