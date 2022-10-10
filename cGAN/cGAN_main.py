
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-11


< cGAN(Conditional Generative Adversarial Networks) ���� >

 - Conditional Generative Adversarial Nets (Arxiv 2014)


< Condition Vector>

 - ������ GAN �� ������ ��� ���������� �Ʒ��� ��ġ�� Condition Vector�� �߰�(Concatenation)

    * Generator�� �Է�

    * Dircriminator�� �Է� ( bath real_img and fake_img )


 - MNIST data�� Ȱ��������  0 ~ 9 �� ���� one-hot vector�� condition vector�� Ȱ�� 
 
   => ���� GAN �𵨿����� ������ �δ��κ��� �Ļ��Ǵ� _img�� ����Ͽ�����, cGAN�� ��� _label�� ���

      _label = [128, 1] �̸�, 0~ 9 ���� ���� 

      �̵��� torch.nn.functional.one_hot(_label, num_classes = 10 ) �� ���� ���� one hot vector ȭ 


< Training >

- �н��� ��� label ���� ���Ͽ� noise sample vector(z) +  condition vector ( one hot label vector) �� ������ �н�

  ���ݱ��� z �� label one_hot vector�� concatenation �ϴ� ������ �н��� ����������

  make_image() ������ ���ϴ� condition vector�� ���� �����Ͽ�, condition vector�� ���õ� �̹����� �����Ǵ� ���� Ȯ���غ��� 
      
'''

import torch
import torch.nn as nn
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)

from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torchvision.transforms as trans

from cGAN_model import Generator, Discriminator


# train data format = [size, tensor, normal(mean, std)] 
train_data_form = trans.Compose( [ trans.Resize(28), trans.ToTensor(), trans.Normalize([0.5], [0.5]) ] )

# MNIST DATASET : 60,000 (train), 10,000(test)
# train_data => ������ meta fild ���·� ����
train_data = MNIST( root = './data', train = True, download = True, transform = train_data_form )
# len(data_loader) = 60,000 / 128 = 468.75 
data_loader = torch.utils.data.DataLoader(train_data, batch_size = 128, shuffle = True )

# set hyper params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learing_rate = 0.0005
batch_size = 128 
n_epoch = 500

# cGAN �� ���� params
num_class = 10 
 # �н��� cGAN�� �̿��Ͽ� target_batch �� ��ŭ ���ϴ� �̹����� �׷����� 
target_batch = 10  
target_num = None

# �����ڿ��� ������ noise sample�� ���� �� ����� ������ ũ�� 
laten_input_dim = 100   

# set NN instance
generator = Generator(laten_input_dim + num_class ).to(device)
discriminator = Discriminator().to(device)

# set loss
criterion = nn.BCELoss().to(device)

# set optim
ge_optimizer = optim.Adam(generator.parameters(), lr = learing_rate)
dis_optimizer = optim.Adam(discriminator.parameters(), lr = learing_rate) 

# training
for epoch in range(n_epoch):

    for i , (_img, _label) in enumerate(data_loader):

        # _img dim = [128,1,28,28]
        _img, _label = _img.to(device), _label.to(device)

        # gt : ground_truth , dim = [128, 1]
        gt_real = torch.cuda.FloatTensor(_img.shape[0],1).fill_(1.0)
        gt_fake = torch.cuda.FloatTensor(_img.shape[0],1).fill_(0.0)

        
        ## train - generator
        ge_optimizer.zero_grad()

        # random noise sampling , z_dim = [128, 100]
        z = torch.normal(mean=0, std=1, size = (_img.shape[0], laten_input_dim)).to(device)
        
        # conditional vector  c := conditional class , dim = [128, 10]
        c = torch.nn.functional.one_hot(_label, num_classes = 10 ).to(device)

        # conditional input = [128, 100+10]
        conditional_z = torch.cat( (z,c), dim = 1 )

        # _generated_img_dim = [128,1,28,28]
        _generated_img = generator( conditional_z )
        
        # log(1-D(G(z))) 
        g_loss = criterion(discriminator(_generated_img, c), gt_real)

        # update
        g_loss.backward()
        ge_optimizer.step()


        ## train - discriminator
        dis_optimizer.zero_grad()

        # E[ log(D(real_x)) ] + E[ log(1-D(G(z))) ]
        real_loss = criterion(discriminator(_img, c), gt_real)
        # _generated_img.detach() : gradient ���� ���� �ʴ� tensor ����
        fake_loss = criterion(discriminator(_generated_img.detach() , c ), gt_fake)
        dis_loss = (real_loss + fake_loss) / 2

        # update
        dis_loss.backward()
        dis_optimizer.step()


        # �� 50 epoch, 100 iterations ���� GAN�� ���� �����Ǵ� img ���� 
        if epoch % 50 == 0 and i % 100 == 0  :

            # �� 50 epoch , 50 iterator ����  ������ �̹��� 25���� 5 x 5 ���� ���·� ��� 
            save_image(_generated_img[:25], f'./cGAN/train_output/{epoch}_{i}.png', nrow=5, normalize = True)

    print(f'Epoch : {epoch}/{n_epoch}  |  G loss : {g_loss}  | D loss : {dis_loss}')

    if epoch == n_epoch -1 : 
        torch.save(generator.state_dict(), "./cGAN/cGAN.pt")



'''
 �н��� cGAN �� �̿��Ͽ� ���ϴ� target �� ���� �̹��� �����ϱ� 

'''
def make_image():

    # load model
    generator.load_state_dict(torch.load("./cGAN/cGAN.pt"))

    with torch.no_grad():

        # random noise latent vector
        z = torch.normal(mean=0, std=1, size = (target_batch, laten_input_dim)).to(device)
        
        print("MNIST dataset �߿��� ����� ���� ���ڸ� �����ϼ��� ( 0 ~ 9 ) : ")
        target_num = int(input())

        # condition vector
        c = torch.tensor( [ target_num for _ in range(target_batch) ] )

        c = torch.nn.functional.one_hot(c, num_classes = 10 ).to(device)

        condition_input = torch.cat( (z,c) , dim = 1)

        output = generator( condition_input )

        save_image(output, f'./cGAN/custom_output/you_want_{target_num}_images.png', nrow = 5, normalize = True )

        print("Close cGAN ")

make_image()
