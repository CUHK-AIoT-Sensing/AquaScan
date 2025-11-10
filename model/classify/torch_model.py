from torchvision.models import resnet34
from torch import nn
import torch
# from BaseVAE import BaseVAE
from torch.nn import functional as F
from .types_ import *
from sklearn.neighbors import NearestNeighbors
# from resnet_uni import model_dict

class Res34Torch(nn.Module):
    def __init__(self, num_classes=5, in_channel=1):
        super(Res34Torch, self).__init__()
        model = resnet34(pretrained=False)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(in_features=fc_features, out_features=num_classes)
        model.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.module = model

    def forward(self, x):
        return self.module(x)
    
    