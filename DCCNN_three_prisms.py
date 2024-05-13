## This program is designed for calculating the potential field downward continuation.
## Users can change the experimental data according to their own needs.

import numpy as np
import scipy.io as scio
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch import nn
from torch.nn import MSELoss


## Transform the .mat file into Tensor
def mat_to_Tensor(path: str, data_name: str) -> Tensor:
    mat_dict = scio.loadmat(path)
    mat_data = mat_dict[data_name]
    mat_data = torch.tensor(mat_data)
    return mat_data

## compute Rmse
def Rmse_compute(x,y,nx,ny):
    Rmse1 = (x-y)**2
    Rmse2 = Rmse1.sum()
    Rmse_result = torch.sqrt(1/(nx * ny) * Rmse2)
    return Rmse_result

## compute Re
def Re_compute(x,y):
    Re_result = (torch.linalg.norm(x[0,0,:,:]-y[0,0,:,:],2)/torch.linalg.norm(y[0,0,:,:],2))*100
    return Re_result


def corr2d(X, K, device):
    X = X.to(device)
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    Y = Y.to(device)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


## calculate the optimal learning rate
def adjust_learning_rate(optimizer, dot_vt_grad, Agradient_2norm):
    lr = dot_vt_grad / Agradient_2norm
    lr = lr.item()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

device = torch.device("cuda:0")

## 'inputs' is the matrix K
inputs = mat_to_Tensor('kernel_3prisms.mat','kernel_3prisms')

## 'targets' is the observation data
targets = mat_to_Tensor('data_3prisms_2.5z.mat','data_3prisms_2point5z')# observation data (with 2.5% noise)
# targets = mat_to_Tensor('data_3prisms_5z.mat','data_3prisms_5z')# observation data (with 5% noise)

## 'kernel1' is the theoretical data at z = 1 km
kernel1 = mat_to_Tensor('three_prism_1000m.mat','three_prism_1000m')

## initialization of convolution kernel
initial_guess = targets


nx = targets.size()[0]
ny = targets.size()[1]
kernel1 = torch.reshape(kernel1,(1,1,nx,ny))
kernel1 = torch.rot90(kernel1,k=-2, dims = [2,3])
kernel1 = kernel1.to(device)  # turn to GPU
initial_guess = initial_guess.to(device)  # turn to GPU

inputs = torch.reshape(inputs,(1,1,2*nx-1,2*ny-1))
targets = torch.reshape(targets,(1,1,nx,ny))
initial_guess = torch.reshape(initial_guess,(1,1,nx,ny))

class DCCNN(nn.Module):
    def __init__(self):
        super(DCCNN,self).__init__()
        self.model = nn.Conv2d(1, 1, (nx, ny), 1, padding='valid', bias=False)

    def forward(self,x):
        x = self.model(x)
        return x


dccnn = DCCNN()
dccnn.double()
dccnn = dccnn.to(device)
dccnn.model.weight.data=torch.nn.Parameter(torch.rot90(initial_guess,k=-2, dims = [2,3]))  # initialization of the weight
dccnn.model.weight.data = dccnn.model.weight.data.to(device)  # turn to GPU

loss_mse = MSELoss(reduction='sum')
loss_mse = loss_mse.to(device)

lr_value = 0.1  # lr_value can be set to any constant (beacuse lr will be updated in the training process)
momentum_value = 0.95

optimizer = torch.optim.SGD(dccnn.parameters(), lr=lr_value, momentum=momentum_value, nesterov=False)

total_train_step = 0
epoch = 1000
tao = 0.00001
Rmse = []
Re = []
loss_total = []
relative_loss = []
vt = torch.zeros([nx,ny])
vt = vt.to(device)  # turn to GPU
for i in range(epoch):
    print('----------Round {}----------'.format(i + 1))
    print('lr: ',optimizer.state_dict()['param_groups'][0]['lr'])
    dccnn.train()
    inputs = inputs.to(device)  # turn to GPU
    targets = targets.to(device)  # turn to GPU
    outputs = dccnn(inputs)
    loss = 0.5 * loss_mse(outputs, targets)
    with torch.no_grad():
        loss_total.append(loss.item())
        relative_loss.append(abs(loss_total[i - 1] - loss_total[i]) / loss_total[i - 1])
        if i > 0 and relative_loss[i] < tao:  # stopping criterion
            break
    optimizer.zero_grad()
    loss.backward()

    with torch.no_grad():
        grad1 = torch.reshape(dccnn.model.weight.grad, (nx, ny))
        vt = momentum_value * vt + grad1
        dot_vt_grad = torch.sum(vt * grad1)
        inputs1 = torch.reshape(inputs, (2*nx-1,2*ny-1))
        Agradient = corr2d(inputs1,vt,device)
        Agradient_2norm = torch.linalg.norm(Agradient)
        Agradient_2norm = Agradient_2norm ** 2

    adjust_learning_rate(optimizer, dot_vt_grad, Agradient_2norm)

    optimizer.step()

    with torch.no_grad():
        total_train_step = total_train_step + 1
        Rmse.append(Rmse_compute(dccnn.model.weight, kernel1, nx, ny).item())
        Re.append(Re_compute(dccnn.model.weight, kernel1).item())

        print('iteration number: {}, Loss: {}'.format(total_train_step, loss.item()))
        print('relative_loss: {}'.format(relative_loss[i]))
        print('Rmse: {}'.format(Rmse[i]))
        print('Re: {}'.format(Re[i]))

loss_total = torch.tensor(loss_total)
Rmse = torch.tensor(Rmse)
Re = torch.tensor(Re)
relative_loss = torch.tensor(relative_loss)
print('Rmse: {}'.format(Rmse_compute(dccnn.model.weight, kernel1, nx, ny)))
print('Re: {}'.format(Re_compute(dccnn.model.weight, kernel1)))

dc_result= torch.rot90(dccnn.model.weight, k=-2, dims = [2,3])
dc_result = torch.reshape(dc_result,(nx,ny))  # downward continuation result