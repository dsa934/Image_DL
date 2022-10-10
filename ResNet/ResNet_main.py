
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-04


< ResNet ���� >

 - ResNet: Deep Residual Learning for Image Recognition


< Shortcut(Skip) Connections > 

 - H(x) : = F(x) + x 

   H(x) : ���������� �ǵ��ϴ� optimal mapping ( �������� NN�� �̿��Ͽ� ���������� ��� ������ �Լ��� �н��ϱ� ���� )


         x             => �Է� ������ x�� Non-linear �� NN�� ��ġ�鼭, H(x)�� ��� Ư¡(�ǵ�)�� �н��Ǳ⸦ �������� �̴� ��ǻ� ��ƴ�
         ��
     Weight Layer         NN�� depth�� ���������, �ش� layer���� �н��Ǵ� Ư¡���� �ٿ��� ������ �� �ֱ� ����(����)
         ��
        ReLU              �ش��� ���÷�, �ǵ��ϴ� H(x) = Identity mapping �� ���, H(x) = x ������  F(x) -> 0 �� �ǵ��� �н��� �����ϸ� �ȴ�
         ��
     Weight Layer         �̷��� �������� �����ϸ�, H(x)�� �н��ϴ� �ͺ��ٴ� F(x) + x �� �н��ϴ� ���� ��, �ܼ��� ���ϴ� ���� ������
         ��
        H(x)              Parameter�� ��, ��� ���⵵�� �߰������� �������� �ʰ�, �н��� ���̵��� �������� ������ �ִ�.
         
                          ����,  optimal mapping = identity mapping �� Ȯ�� ��ü�� ������, 

                          optimal mapping �� zero mapping�� �ƴ�, identity mapping �̶�� Shortcut connections�� �ſ� ȿ������ �� �ִ�.


'''

import torch
import torch.nn as nn 
import torch.optim as optim

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as trans

from ResNet_model import ResNet

# set hyper params
n_epoch = 10 
batch_size = 128
num_class = 10
learning_rate = 5e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )


train_form = trans.Compose([trans.ToTensor()])
test_form = trans.Compose([trans.ToTensor()])


train_data = MNIST(root='./data', train = True, download = True, transform = train_form )
test_data = MNIST(root='./data', train = False, download = True, transform = test_form)

# train_loader = 60,000 // 128 = 468.x
# test_loader = 10,000 // 128 = 78.125
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True)


model = ResNet(num_class).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9, weight_decay = 2e-4)


def train(epoch):

    print(f"Train Epoch : {epoch}")

    model.train()

    train_loss, correct, total = 0, 0, 0


    for batch_idx, (_train, _label) in enumerate(train_loader):

        # _train = [128, 1, 28, 28] , _label = [128, 1]
        _train, _label = _train.to(device), _label.to(device)

        # optimizer init
        optimizer.zero_grad()

        model_output = model(_train)

        # cal loss
        loss = criterion(model_output, _label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
 
        # [128,10] , 10 ���� �ĺ��� ���� ū �� 
        _, pred_val = model_output.max(1)

        total += _label.shape[0]
        correct += pred_val.eq(_label).sum().item()

        if batch_idx % 100 == 0 :
            
            print(f"Current Batch :{batch_idx}")
            print(f"Current train accuracy : {pred_val.eq(_label).sum().item() / _label.shape[0]}")
            print(f"Current each batch's 'train loss : {loss.item()}\n")

    print(f"Total Train Accuracy : {100.0 * correct /total}")
    print(f"Total Train Loss : {train_loss}\n")
    

def test(epoch):

    print(f"Test Epoch {epoch}")

    model.eval()

    test_loss, correct, total = 0, 0, 0

    for batch_idx, (_test, _label) in enumerate(test_loader):

        _test, _label = _test.to(device), _label.to(device)

        total += _label.shape[0]

        test_output = model(_test)

        test_loss += criterion(test_output, _label).item()

        # test_output = [128, 10] , 10�� �� ���� ���� �ĺ� ����
        _, test_pred = test_output.max(1)

        correct += test_pred.eq(_label).sum().item()


    print(f"Test Accuracy : {100.0* correct / total}")
    print(f"Test Avg Loss : {test_loss / total }\n")



for e_idx in range(n_epoch):

    train(e_idx)
    test(e_idx)