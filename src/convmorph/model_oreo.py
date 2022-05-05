from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ConvmorphOreoModelOutput:
  loss: torch.tensor
  pred_vec: torch.tensor

class ConvmorphOreoModel(nn.Module):
  def __init__(self, dropout=0.1, init_emb=None, n_vocab=5000, h_dim=200):
    super().__init__()

    if init_emb is not None:
      self.emb = nn.Embedding.from_pretrained(init_emb, freeze=False)
    else:
      self.emb = nn.Embedding(n_vocab, h_dim)
    
    self.h_dim = h_dim
    self.fn1 = nn.Linear(h_dim*2, h_dim*8)
    self.norm1 = nn.LayerNorm(h_dim*8)
    self.fn2 = nn.Linear(h_dim*8, h_dim*8)
    self.norm2 = nn.LayerNorm(h_dim*8)
    # another layer
    self.fn3 = nn.Linear(h_dim*8, h_dim*8)
    self.norm3 = nn.LayerNorm(h_dim*8)

    self.drop1 = nn.Dropout(p=dropout)    
    self.fn4 = nn.Linear(h_dim*8, 200)
  
  def forward(self, inputX, target=None, **kwargs):
    
    z = self.emb(inputX).view(-1, self.h_dim*2)
    z = torch.tanh(self.norm1(self.fn1(z)))
    z = torch.tanh(self.norm2(self.fn2(z)))
    z = torch.tanh(self.norm3(self.fn3(z)))
    z = self.drop1(z) 
    o = self.fn4(z)
    
    pred_vec = o

    if target is not None:
      loss_fct = nn.MSELoss()
      loss = loss_fct(pred_vec, target)
      return ConvmorphOreoModelOutput(loss, pred_vec)
    else:      
      return ConvmorphOreoModelOutput(float('nan'), pred_vec)
    