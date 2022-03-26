import torch
import torch.nn as nn
import torch.nn.functional as F
class SeismicNet(nn.Module):
    def __init__(self,):
        super(SeismicNet, self).__init__()

#全卷积
#model2 原始数据1024维
        self.features=nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2,bias=True),#nx16x510
            nn.Conv1d(16, 16, kernel_size=1, stride=1,bias=True),#nx16x510
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2),#nx16x254
            nn.Conv1d(16, 32, kernel_size=5, stride=2,bias=True),#nx32x125
            nn.Conv1d(32, 32, kernel_size=1, stride=1,bias=True),#nx32x125
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2),#nx32x62
            nn.Conv1d(32, 48, kernel_size=5, stride=2,bias=True),#nx48x29
            nn.Conv1d(48, 48, kernel_size=1, stride=1,bias=True),#nx48x29
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2),#nx48x14
            nn.Conv1d(48, 64, kernel_size=3, stride=2,bias=True),#nx64x6
            nn.Conv1d(64, 64, kernel_size=1, stride=1,bias=True),#nx64x6
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=1),#nx64x4
            nn.Conv1d(64, 3, kernel_size=4, stride=1,bias=True),#nxclass_numx1
            )
 
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        #print(x)
        x = x.view(x.size(0), -1)
        #x=self.classifier(x)
        #print(x)
        x = F.softmax(x,dim=1)
        return x 
