import torch
import torch.nn as nn

class WaveletAcoModel(nn.Module):
    def __init__(self):
        super(WaveletAcoModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.Linear(64, 3),
            nn.Sigmoid()
            )
    def forward(self,x):
        x =self.network(x)
        return x
