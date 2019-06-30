from torch import nn
import torch
import numpy as np
import string
import torch.nn.functional as F
import torch.autograd as autograd
from sru import SRU, SRUCell

class cnn(nn.Module):
    def __init__(self, input_dim, n_class):
        super(cnn, self).__init__()
        vocb_size = input_dim
        self.dim = 100
        self.max_len = 20000
        self.embeding = nn.Embedding(vocb_size, self.dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim, out_channels=16, kernel_size=5,
                      stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.max_len)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim, out_channels=16, kernel_size=4,
                      stride=1, padding=2),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.max_len)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim, out_channels=16, kernel_size=3,
                      stride=1, padding=2),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.max_len)
        )
        self.out = nn.Linear(48, n_class)


    def forward(self, index):
        x = self.embeding(index)
        x = x.unsqueeze(3)
        x = x.permute(0, 2, 1, 3)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = x.view(-1, x.size(1))
        output = self.out(x)
        return {"pred": output}


class rnn(nn.Module):
    def __init__(self, input_size, n_class):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=100,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.embedding = nn.Embedding(input_size, 100)
        self.out = nn.Linear(32, n_class)

    def forward(self, index):
        data = self.embedding(index)
        output, hidden = self.rnn(data)
        output = self.out(output)
        # 仅仅获取 time seq 维度中的最后一个向量
        output = torch.mean(output, dim=1, keepdim=True)
        return {"pred": output[:,-1,:]}


class myLSTM(nn.Module):
    def __init__(self, input_size, n_class, batch=1):
        super().__init__()
        self.hidden_dim = 128
        self.myLSTM = torch.nn.LSTM(
            input_size=100,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.avg1d = nn.AvgPool1d(20000)
        # self.hidden = (torch.zeros(1, batch, self.hidden_dim),
        #                 torch.zeros(1, batch, self.hidden_dim))
        self.embedding = nn.Embedding(input_size, 100)
        self.out = nn.Linear(self.hidden_dim, n_class)
	
    def forward(self, index):
        x = self.embedding(index)
        output, hidden = self.myLSTM(x)
        x = output.permute(0, 2 ,1)
        x = self.avg1d(x)
        x = x.view(-1, x.size(1))
        x = self.out(x)
        return {"pred": x}



class myQnn(nn.Module):
    def __init__(self, input_dim, n_class):
        super(myQnn, self).__init__()
        vocb_size = input_dim
        self.hidden_dim = 128
        self.dim = 100
        self.max_len = 20000
        self.embedding = nn.Embedding(vocb_size, self.dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2*self.hidden_dim, out_channels=16, kernel_size=5,
                      stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.max_len)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=2*self.hidden_dim, out_channels=16, kernel_size=4,
                      stride=1, padding=2),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.max_len)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=2*self.hidden_dim, out_channels=16, kernel_size=3,
                      stride=1, padding=2),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.max_len)
        )
        self.myLSTM = SRU(
            input_size=100,
            hidden_size=128,
            num_layers=1,
        )
        self.avg1d = nn.AvgPool1d(self.max_len)
        self.out = nn.Linear(48, n_class)


    def forward(self, index):
        y = index
        for row in range(index.shape[0]):
            for j in range(index.shape[1]):
                y[row][j] = index[row][(index.shape[1])-j-1]

        # x = self.embedding(index)
        # x_rev = y.detach().numpy()[:,: ,::-1].copy()
        # #output1, hidden1 = self.myLSTM(x)
        # output2, hidden2 = self.myLSTM(torch.tensor(x_rev)) #反转
        # # x = torch.cat((output1, output1), dim=2) # 在特征层维度进行扩展
        # #x = output1
        x = self.embedding(index)
         # Reverse of copy of numpy array of given tensor
        #y = y.cpu().detach().numpy()[::-1]
        y = self.embedding(y)
        output1, hidden1 = self.myLSTM(x)
        output2, hidden2 = self.myLSTM(y)  # 反转
        x = torch.cat((output1, output2), dim=2)
        x = x.permute(0, 2 ,1)
        x = self.avg1d(x)
        x = x.unsqueeze(3)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = x.view(-1, x.size(1))
        output = self.out(x)
        return {"pred": output}
