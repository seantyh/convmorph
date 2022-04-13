from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from .convmorph_dataset import ConvmorphDataset

def make_linear_projection(ds: ConvmorphDataset, train_idxs):
    trainA = np.vstack([
        np.concatenate([
            ds[idx]["const1_vec"], 
            ds[idx]["const2_vec"]]) 
        for idx in train_idxs
        ])
    trainB = np.vstack([ds[idx]["word_vec"] for idx in train_idxs])
    AtA_inv = np.linalg.inv(np.dot(trainA.transpose(), trainA))
    trainX = np.dot(np.dot(AtA_inv, trainA.transpose()), trainB)    
    return trainX

def make_random_projection(ds: ConvmorphDataset, train_idxs, full_random=False):
    rng = np.random.RandomState(123)
    if full_random:
        trainA = rng.standard_normal((len(train_idxs), ds[0]["word_vec"].shape[0]*2))
    else:
        trainA = np.vstack([
            np.concatenate([
                ds[idx]["const1_vec"], 
                ds[idx]["const2_vec"]]) 
            for idx in train_idxs
            ])
    
    randomB = rng.standard_normal((len(train_idxs), ds[0]["word_vec"].shape[0]))
    AtA_inv = np.linalg.inv(np.dot(trainA.transpose(), trainA))
    randomX = np.dot(np.dot(AtA_inv, trainA.transpose()), randomB)    
    return randomX

@dataclass
class LinearProjectionOutput:  
    pred_vec: torch.tensor

class LinearProjection(nn.Module):
    def __init__(self, proj, device):
        super().__init__()
        self.proj = torch.tensor(proj, dtype=torch.float32).to(device)
  
    def forward(self, inputX, **kwargs):
        inputX = inputX.view(-1, 400)
        return LinearProjectionOutput(torch.matmul(inputX, self.proj))
    