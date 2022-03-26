#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 12:49:33 2020

@author: sunlin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
import numpy as np
from torch import Tensor
import math
class SEISClassifierLSTM(nn.Module):
    #声明带有模型参数的层，这里声明了四层
  
    def __init__(self,lstm_inputsize,lstm_hiddensize,lstm_numlayers):
        super(SEISClassifierLSTM, self).__init__()
        #假设数据格式不正常
        self.conv = nn.Sequential(
            #cnn1hp.signal.wavelet_energyfeatures
            nn.Conv1d(1, 8, kernel_size=5, stride=2,bias=True),#nx48x29
            nn.Conv1d(8, 16, kernel_size=1, stride=1,bias=True),#nx48x29
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2),#nx48x14
            nn.Conv1d(16, 32, kernel_size=3, stride=2,bias=True),#nx64x6
            nn.Conv1d(32, 32, kernel_size=1, stride=1,bias=True),#nx64x6
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=1),#nx64x4
            nn.Conv1d(32, 8, kernel_size=4, stride=1,bias=True),#nxclass_numx1
            nn.Conv1d(8, 1, kernel_size=4, stride=1,bias=True),#nxclass_numx1
            )
        self.rnn = nn.LSTM(lstm_inputsize,lstm_hiddensize,lstm_numlayers,bias=True,bidirectional=True)
        self.fc1 = nn.Sequential(
            nn.Linear(lstm_hiddensize*4,64),
            #nn.BatchNorm1d(64),nn.ReLU(),
            nn.Linear(64,3),
            #nn.BatchNorm1d(3),
            #nn.Sigmoid(),
        ) 
    #定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(2,0,1)
        #print(x.permute(1,0,2).shape)
        output,_ = self.rnn(x)
        #print(output.shape)
        output = torch.cat((output[0],output[-1]),-1)
        #print(output.shape)
        #output = output.permute(1,0,2)
        #output = output[:,-1,:]
        #print(output.shape)
        output = self.fc1(output)
        return output


if __name__=="__main__":
    class LSTM():
        def __init__(self):
            self.input_size=1
            self.hidden_size=128
            self.num_layers=2
    class HP():
        def __init__(self):
            self.lstm=LSTM()
            
    hp=HP()
    model=SEISClassifierLSTM(hp)
    x = torch.rand(2,1024)
    output = model(x)
    print( 'out:', output)
