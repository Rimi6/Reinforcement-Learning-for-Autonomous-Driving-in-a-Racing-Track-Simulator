# Import Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import time
from collections import deque


class SecondCNNForPPO(nn.Module):
    def __init__(self, p_oConfig):
        super(SecondCNNForPPO, self).__init__()

        self.Config = p_oConfig
        self.img_stack = self.Config["Data.ImageStackCount"]

        self.InputLayer1 = self.Config["InputLayer1"]
        self.InputLayer2 = self.Config["InputLayer2"]
        
        self.OutputLayer0 = self.Config["OutputLayer0"]
        self.OutputLayer1 = self.Config["OutputLayer1"]
        self.OutputLayer2 = self.Config["OutputLayer2"]

        

        self.conv = nn.Sequential(
            nn.Conv2d(self.img_stack, self.OutputLayer0, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(self.InputLayer1, self.OutputLayer1, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(self.InputLayer2, self.OutputLayer2, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Linear(4096, 512)

        self.alpha_head = nn.Sequential(
            nn.Linear(512, 3), 
            nn.Softplus()
        )
        
        self.beta_head = nn.Sequential(
            nn.Linear(512, 3), 
            nn.Softplus()
        )

        self.v = nn.Sequential(
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc(x))

        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1
        v = self.v(x)

        return (alpha, beta), v