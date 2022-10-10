
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-11


< cGAN(Conditional Generative Adversarial Networks) 복원 >

 - Conditional Generative Adversarial Nets (Arxiv 2014)


< Condition Vector>

 - 기존의 GAN 과 구조는 모두 동일하지만 아래의 위치에 Condition Vector만 추가(Concatenation)

    * Generator의 입력

    * Dircriminator의 입력 ( bath real_img and fake_img )


 - MNIST data를 활용함으로  0 ~ 9 에 대한 one-hot vector를 condition vector로 활용 
 
   => 기존 GAN 모델에서는 데이터 로더로부터 파생되는 _img만 사용하였지만, cGAN의 경우 _label도 사용

      _label = [128, 1] 이며, 0~ 9 값이 분포 

      이들을 torch.nn.functional.one_hot(_label, num_classes = 10 ) 을 통해 각각 one hot vector 화 


< Training >

- 학습은 모든 label 값에 대하여 noise sample vector(z) +  condition vector ( one hot label vector) 의 구조로 학습

  지금까지 z 에 label one_hot vector를 concatenation 하는 식으로 학습을 진행했으니

  make_image() 에서는 원하는 condition vector를 직접 기입하여, condition vector에 관련된 이미지만 생성되는 것을 확인해보기 
      
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
# train_data => 일종의 meta fild 형태로 저장
train_data = MNIST( root = './data', train = True, download = True, transform = train_data_form )
# len(data_loader) = 60,000 / 128 = 468.75 
data_loader = torch.utils.data.DataLoader(train_data, batch_size = 128, shuffle = True )

# set hyper params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learing_rate = 0.0005
batch_size = 128 
n_epoch = 500

# cGAN 을 위한 params
num_class = 10 
 # 학습된 cGAN을 이용하여 target_batch 수 만큼 원하는 이미지를 그려볼것 
target_batch = 10  
target_num = None

# 생성자에서 임의의 noise sample을 뽑을 떄 사용할 벡터의 크기 
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
        # _generated_img.detach() : gradient 전파 되지 않는 tensor 복제
        fake_loss = criterion(discriminator(_generated_img.detach() , c ), gt_fake)
        dis_loss = (real_loss + fake_loss) / 2

        # update
        dis_loss.backward()
        dis_optimizer.step()


        # 매 50 epoch, 100 iterations 마다 GAN에 의해 생성되는 img 저장 
        if epoch % 50 == 0 and i % 100 == 0  :

            # 매 50 epoch , 50 iterator 마다  생성된 이미지 25장을 5 x 5 격자 형태로 출력 
            save_image(_generated_img[:25], f'./cGAN/train_output/{epoch}_{i}.png', nrow=5, normalize = True)

    print(f'Epoch : {epoch}/{n_epoch}  |  G loss : {g_loss}  | D loss : {dis_loss}')

    if epoch == n_epoch -1 : 
        torch.save(generator.state_dict(), "./cGAN/cGAN.pt")



'''
 학습된 cGAN 을 이용하여 원하는 target 에 대한 이미지 생성하기 

'''
def make_image():

    # load model
    generator.load_state_dict(torch.load("./cGAN/cGAN.pt"))

    with torch.no_grad():

        # random noise latent vector
        z = torch.normal(mean=0, std=1, size = (target_batch, laten_input_dim)).to(device)
        
        print("MNIST dataset 중에서 만들고 싶은 숫자를 선택하세요 ( 0 ~ 9 ) : ")
        target_num = int(input())

        # condition vector
        c = torch.tensor( [ target_num for _ in range(target_batch) ] )

        c = torch.nn.functional.one_hot(c, num_classes = 10 ).to(device)

        condition_input = torch.cat( (z,c) , dim = 1)

        output = generator( condition_input )

        save_image(output, f'./cGAN/custom_output/you_want_{target_num}_images.png', nrow = 5, normalize = True )

        print("Close cGAN ")

make_image()
