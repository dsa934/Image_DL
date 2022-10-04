
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-10-04


< ResNet 복원 >

 - ResNet: Deep Residual Learning for Image Recognition


< Shortcut(Skip) Connections > 

 - H(x) : = F(x) + x 

   H(x) : 본질적으로 의도하는 optimal mapping ( 여러개의 NN을 이용하여 점진적으로 어떠한 복잡한 함수를 학습하길 원함 )


         x             => 입력 데이터 x가 Non-linear 한 NN을 거치면서, H(x)에 어떠한 특징(의도)이 학습되기를 원하지만 이는 사실상 어렵다
         ↓
     Weight Layer         NN의 depth가 깊어질수록, 해당 layer에서 학습되는 특징들의 근원이 잊혀질 수 있기 떄문(추측)
         ↓
        ReLU              극단적 예시로, 의도하는 H(x) = Identity mapping 일 경우, H(x) = x 임으로  F(x) -> 0 이 되도록 학습을 진행하면 된다
         ↓
     Weight Layer         이러한 관점에서 접근하면, H(x)를 학습하는 것보다는 F(x) + x 를 학습하는 형태 즉, 단순히 더하는 형태 임으로
         ↓
        H(x)              Parameter의 수, 계산 복잡도가 추가적으로 증가하지 않고, 학습의 난이도가 쉬워지는 장점이 있다.
         
                          따라서,  optimal mapping = identity mapping 일 확률 자체는 적지만, 

                          optimal mapping 이 zero mapping이 아닌, identity mapping 이라면 Shortcut connections은 매우 효과적일 수 있다.


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
 
        # [128,10] , 10 개의 후보중 가장 큰 값 
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

        # test_output = [128, 10] , 10개 중 가장 높은 후보 추출
        _, test_pred = test_output.max(1)

        correct += test_pred.eq(_label).sum().item()


    print(f"Test Accuracy : {100.0* correct / total}")
    print(f"Test Avg Loss : {test_loss / total }\n")



for e_idx in range(n_epoch):

    train(e_idx)
    test(e_idx)