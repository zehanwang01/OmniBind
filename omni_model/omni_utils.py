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
from paths import *
from huggingface_hub import hf_hub_download, snapshot_download

def seg_mean(x, split_size_or_sections=7, dim=1):
    x = torch.split(x, split_size_or_sections=split_size_or_sections, dim=dim)
    x = [seg.mean(dim=dim, keepdim=True) for seg in x]
    x = torch.cat(x, dim=dim) # emb: b*enc_nums*emb_dim | weight: b*enc_nums*1
    return x

def get_config(mode):
    if mode == 'Full':
        config = {}
        AT_ckpt_modes = ['CLAP_G',
                         'CLAP_M',
                         'WAVCAPS_PT',
                         'WAVCAPS_FT_aud',
                         'WAVCAPS_FT_clo']
        VT_ckpt_modes = ['DFN_H14',
                         'EVA_E14p',
                         'SigLip_384',
                         'SigLip_400M_384']
        PVT_ckpt_modes = ['UNI3D_L',
                          'UNI3D_M',
                          'UNI3D_S']
        config['EVA_18B'] = True
        config['A'] = AT_ckpt_modes + ['IB', 'EVA_18B']
        config['V'] = VT_ckpt_modes + PVT_ckpt_modes + ['IB', 'EVA_18B']
        config['T'] = AT_ckpt_modes + VT_ckpt_modes + PVT_ckpt_modes + ['IB', 'EVA_18B']
        config['P'] = PVT_ckpt_modes
        config['all'] = AT_ckpt_modes + VT_ckpt_modes + PVT_ckpt_modes + ['IB', 'EVA_18B']
        config['A_pred'] = 2
        config['V_pred'] = 4
        config['P_pred'] = 0
        config['T_pred'] = [2,9,10]
        config['A_pred_dim'] = 1024
        config['V_pred_dim'] = 1152
        config['P_pred_dim'] = 1024
        config['T_pred_dim'] = 1024 + 1152 + 1024
        config['A_num'] = 5
        config['V_num'] = 4
        config['P_num'] = 3
        
        config['in_dim'] = {'AT':[512, 512, 1024, 1024, 1024], 'VT': [1024, 1024, 1024, 1152], 'PVT': [1024, 1024, 1024]}
    if mode == 'Large':
        config = {}
        AT_ckpt_modes = ['CLAP_G',
                         'CLAP_M',
                         'WAVCAPS_PT',
                         'WAVCAPS_FT_aud',
                         'WAVCAPS_FT_clo']
        VT_ckpt_modes = ['DFN_H14',
                         'EVA_E14p',
                         'SigLip_384',
                         'SigLip_400M_384']
        PVT_ckpt_modes = ['UNI3D_L',
                          'UNI3D_M',
                          'UNI3D_S']
        config['EVA_18B'] = False
        config['A'] = AT_ckpt_modes + ['IB']
        config['V'] = VT_ckpt_modes + PVT_ckpt_modes + ['IB']
        config['T'] = AT_ckpt_modes + VT_ckpt_modes + PVT_ckpt_modes + ['IB']
        config['P'] = PVT_ckpt_modes
        config['all'] = AT_ckpt_modes + VT_ckpt_modes + PVT_ckpt_modes + ['IB']
        config['A_pred'] = 2
        config['V_pred'] = 3
        config['P_pred'] = 0
        config['T_pred'] = [2,8,9]
        config['A_pred_dim'] = 1024
        config['V_pred_dim'] = 1152
        config['P_pred_dim'] = 1024
        config['T_pred_dim'] = 1024 + 1152 + 1024
        config['A_num'] = 5
        config['V_num'] = 4
        config['P_num'] = 3
        
        config['in_dim'] = {'AT':[512, 512, 1024, 1024, 1024], 'VT': [1024, 1024, 1024, 1152], 'PVT': [1024, 1024, 1024]}
    if mode == 'Base':
        config = {}
        AT_ckpt_modes = ['WAVCAPS_PT',
                         'WAVCAPS_FT_aud',
                         'WAVCAPS_FT_clo']
        VT_ckpt_modes = ['EVA_E14p']
        PVT_ckpt_modes = ['UNI3D_L']
        config['EVA_18B'] = False
        config['A'] = AT_ckpt_modes + ['IB']
        config['V'] = VT_ckpt_modes + PVT_ckpt_modes + ['IB']
        config['T'] = AT_ckpt_modes + VT_ckpt_modes + PVT_ckpt_modes + ['IB']
        config['P'] = PVT_ckpt_modes
        config['all'] = AT_ckpt_modes + VT_ckpt_modes + PVT_ckpt_modes + ['IB']
        config['A_pred'] = 0
        config['V_pred'] = 0
        config['P_pred'] = 0
        config['T_pred'] = [0,3,4]
        config['A_pred_dim'] = 1024
        config['V_pred_dim'] = 1024
        config['P_pred_dim'] = 1024
        config['T_pred_dim'] = 1024 + 1024 + 1024
        config['A_num'] = 3
        config['V_num'] = 1
        config['P_num'] = 1
        
        config['in_dim'] = {'AT':[1024, 1024, 1024], 'VT': [1024], 'PVT': [1024]}
    
    return config


def check_download():
    if not os.path.exists('./checkpoints'):
        snapshot_download(repo_id="Viglong/OmniBind", local_dir=".")
    else:
        print('checkpoints dir already exists')