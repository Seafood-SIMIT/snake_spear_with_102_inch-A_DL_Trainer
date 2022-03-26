import torch
import torch.nn as nn
class UrbanSoundModel(nn.Module):
    def __init__(self):
        super(UrbanSoundModel,self).__init__()
        self.network = nn.Sequential(
            nn.Linear(32,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.Linear(64, 3),
            nn.Sigmoid()
            )
    def forward(self, xb):
        #xb = torch.tensor(xb, dtype=torch.float32)
        return self.network(xb)
