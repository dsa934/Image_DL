
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-11

 < Dimension >

  * Generator

    => latent_input_dim + label embedding vector = 100 + 10 = 110 

       embedding( 정수 embedding, 원하는 embedding dim) 의 형태로 사용 되며,

       NLP Task 에서는 정수 embedding = vocab_size 가 되지만,

       MNIST data의 경우 MNIST_class(0~9) 에 해당함으로 10이 된다 


  * Discriminator 

    => 28 x 28 image를 flatten 하게 펼침 ( CNN을 사용하지 않기 떄문)

       여기에 condition vector가 추가 됨으로 

       입력 dim = ( 1 * 28 * 28 ) + 10 
 
< 알아두면 좋은 것 >

 - nn.BatchNorm1d( in_Feature, ..)

   => Generator에서 BN을 사용하지 않으면 discriminator's loss가 0으로 수렴 
   
      * 구현에서 nn.Batchnorm(out_feat) 인 이유 
      
         nn.linear(in_feat, out_feat) 

         nn.BatchNorm1d(out_feat)  => batchnorm1d 입장에서는 linear의 output이 batch의 input 임으로 


'''

import torch 
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, img_size, num_class):

        super().__init__()

        # nn.Embedding(정수 임베딩 범위, 사용자 정의 embedding dim)
        self.label_embedding = nn.Embedding(num_class, 10)

        self.model = nn.Sequential(

            nn.Linear( (img_size**2)+ num_class, 512),
            nn.LeakyReLU(0.2, inplace = True ),
            
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace = True ),
            
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace = True ),
            
            nn.Linear(512, 1)
            
            )

    def forward(self, data, label):
        
        label_vector = self.label_embedding(label)

        data = data.view(data.shape[0] , -1 )

        data = torch.cat( (data, label_vector), dim = 1 )

        output = self.model(data)

        return output


class Generator(nn.Module):

    def __init__(self, len_latent, num_class):
        
        super().__init__()

        self.label_embedding = nn.Embedding(num_class, 10)

        # block 
        def g_block(input_dim, output_dim, normalize = True):

            layer = [ nn.Linear(input_dim, output_dim ) ]

            if normalize : 
                # batchnorm1d ( num_feattures, eps, momentum)
                # eps      : a value added to the denominator for numerical stability
                # momentum : the value used for the running_mean and running_var computation
                layer.append( nn.BatchNorm1d(output_dim, momentum = 0.8) )

            layer.append( nn.LeakyReLU(0.2 , inplace = True))


            return layer


        self.model = nn.Sequential(

            *g_block( len_latent + num_class, 128, normalize = False ),
            *g_block(128, 256),
            *g_block(256, 512),
            *g_block(512, 1024),
            nn.Linear(1024, 1*28*28),
            nn.Tanh()

            )


    def forward(self, data, label):
        
        label_vector = self.label_embedding(label)


        data = torch.cat( (data, label_vector), dim = 1 )

        output = self.model(data)

        output = output.view(data.shape[0], 1, 28, 28)


        return output
 