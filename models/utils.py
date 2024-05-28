import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from pathlib import Path
import einops
import math

import torch
import glob
import os

def seg_mean(x, split_size_or_sections=7, dim=1):
    x = torch.split(x, split_size_or_sections=split_size_or_sections, dim=dim)
    x = [seg.mean(dim=dim, keepdim=True) for seg in x]
    x = torch.cat(x, dim=dim) # emb: b*enc_nums*emb_dim | weight: b*enc_nums*1
    return x