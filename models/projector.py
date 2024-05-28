import torch
from torch import nn
import torch.nn.functional as F

import copy

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation.lower() == 'tanh':
        return nn.Tanh()
    else:
        return nn.ReLU(inplace=True)
def get_clones(module: nn.Module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Proj_Head_mini(nn.Module):
    def __init__(self, in_dim, out_dim, init_mode, dim_act='relu'):
        super(Proj_Head_mini, self).__init__()
 
        self.mlps = nn.Sequential(
            nn.Linear(in_dim, 2048, bias=True),
            nn.BatchNorm1d(2048),
            get_activation(dim_act),
            nn.Linear(2048, out_dim, bias=True),
            nn.BatchNorm1d(out_dim),
        )

        self.init_weights(init_mode)

    def forward(self, embs):
        embs = self.mlps(embs)
        return F.normalize(embs, dim=-1)

    def init_weights(self, mode):
        # initialize transformer
        if mode == 'eye':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.eye_(m)
        elif mode == 'xav':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.xavier_uniform_(m)
    
    def get_device(self):
        return next(self.parameters()).device

class Reaction_Head_mini(nn.Module):
    def __init__(self, in_dim=512, out_dim=1024):
        super(Reaction_Head_mini, self).__init__()
        self.Head2 = Proj_Head_mini(in_dim, out_dim, 'xav', 'relu')

    def get_device(self):
        return next(self.parameters()).device
    
    def proj_audio(self, x):
        return self.Head2(x)
    
    def proj_text(self, x):
        return self.Head2(x)
    
    def proj_image(self, x):
        return self.Head2(x)
