import torch
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ConvmorphArcModelOutput:
    loss: torch.tensor
    pred_vec: torch.tensor

class ConvmorphArcModel(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
    
        self.conv1 = nn.Conv2d(1, 128, 3)
        self.norm1 = nn.LayerNorm([128, 18, 18])
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.norm2 = nn.LayerNorm([128, 16, 16])    
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.norm3 = nn.LayerNorm([128, 14, 14])    
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.norm4 = nn.LayerNorm([128, 12, 12])
        self.conv5 = nn.Conv2d(128, 128, 3)
        self.norm5 = nn.LayerNorm([128, 10, 10])
        
        self.pool1 = nn.MaxPool2d(2)
        self.fn1 = nn.Linear(128*5*5, 3000)
        self.drop1 = nn.Dropout(p=dropout)
        self.fn4 = nn.Linear(3000, 200)
        
    def forward(self, inputX, target=None, **kwargs):
        
        z = F.relu(self.norm1(self.conv1(inputX)), inplace=True)
        z = F.relu(self.norm2(self.conv2(z)), inplace=True)
        z = F.relu(self.norm3(self.conv3(z)), inplace=True)
        z = F.relu(self.norm4(self.conv4(z)), inplace=True)    
        z = F.relu(self.norm5(self.conv5(z)), inplace=True)    
        z = self.pool1(z).view(-1, 128*5*5)
        o = self.drop1(torch.tanh(self.fn1(z)))    
        o = self.fn4(o)
        pred_vec = o
    
        if target is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(pred_vec, target)
            return ConvmorphArcModelOutput(loss, pred_vec)
        else:      
            return ConvmorphArcModelOutput(float('nan'), pred_vec)
    