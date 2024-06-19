import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Function
import copy
from tqdm import tqdm
from projector import *
from omni_utils import *
from typing import List, Tuple

class MOE_Router_Sig(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=100, out_dim=2, bias=True):
        super(MOE_Router_Sig, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.router = nn.Sequential(
            nn.Linear(in_dim, 512, bias=bias),
            nn.BatchNorm1d(512),
            get_activation('relu'),
            nn.Linear(512, hidden_dim, bias=bias),
            nn.BatchNorm1d(hidden_dim),
            get_activation('tanh'),
            nn.Linear(hidden_dim, out_dim, bias=bias)
        )
        self.init_weights('xav')
    
    def forward(self, embs):
        res = self.router(embs)
        return F.sigmoid(res)

    def disable_grad(self):
        for param in self.router.parameters():
            param.requires_grad = False
    
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
        elif mode == 'uni':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.uniform_(m)
        elif mode == 'cons':
            constant_value = 1
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.constant_(m, constant_value)
    
    def get_device(self):
        return next(self.parameters()).device

class ATVP_router_UNI(torch.nn.Module):
    def __init__(self, 
                 weights, 
                 config = get_config('Full'),
                 pretrained=True):
        super(ATVP_router_UNI, self).__init__()

        self.A_proj_nums = config['A_num'] * 7
        self.V_proj_nums = config['V_num'] * 7
        self.P_proj_nums = config['P_num'] * 7
        self.T_proj_nums = (config['A_num']+config['V_num']+config['P_num']) * 7
        
        self.A_enc_nums  = config['A_num']+1
        self.V_enc_nums  = config['V_num']+1
        self.P_enc_nums  = config['P_num']
        self.T_enc_nums  = (config['A_num']+config['V_num']+config['P_num'])+1
        
        print(self.A_proj_nums, self.V_proj_nums, self.P_proj_nums, self.T_proj_nums)
        print(self.A_enc_nums , self.V_enc_nums , self.P_enc_nums , self.T_enc_nums)
        
        self.A_enc_router = MOE_Router_Sig(in_dim=config['A_pred_dim'], out_dim=self.A_enc_nums)
        self.V_enc_router = MOE_Router_Sig(in_dim=config['V_pred_dim'], out_dim=self.V_enc_nums)
        self.P_enc_router = MOE_Router_Sig(in_dim=config['P_pred_dim'], out_dim=self.P_enc_nums)
        self.T_enc_router = MOE_Router_Sig(in_dim=config['T_pred_dim'], out_dim=self.T_enc_nums)
        
        self.projs = []
        for i in range(config['A_num']):
            self.projs += [Reaction_Head_mini(in_dim=config['in_dim']['AT'][i], out_dim=1536) for _ in range(7)]
        for i in range(config['V_num']):
            self.projs += [Reaction_Head_mini(in_dim=config['in_dim']['VT'][i], out_dim=1536) for _ in range(7)]
        for i in range(config['P_num']):
            self.projs += [Reaction_Head_mini(in_dim=config['in_dim']['PVT'][i], out_dim=1536) for _ in range(7)]
        self.projs = nn.ModuleList(self.projs)
        
        self.ib_projs = nn.ModuleList([Reaction_Head_mini(in_dim=1024, out_dim=config['out_dim']) for _ in range(7)])
        
        if pretrained:
            self.load_state_dict(torch.load(weights, map_location='cpu'), strict=False)
            
        self.audio_pred = config['A_pred']
        self.image_pred = config['V_pred']
        self.point_pred = config['P_pred']
        self.text_pred  = config['T_pred']
        
    # audio expert + IB[-1]
    def proj_audio(self, seq:List[torch.Tensor]):
        with torch.no_grad():
            ex_audio = [self.projs[i].proj_audio(x[i//7]) for i in range(self.A_proj_nums)] + [self.ib_projs[i].proj_audio(x[-1]) for i in range(7)]
            ex_audio = torch.stack(ex_audio, dim=1)
            ex_audio = seg_mean(ex_audio, 7, dim=1)
            B = len(x[-1])
            # print(ex_audio.shape, self.A_weights.shape)
        A_enc_weights = self.A_enc_router(x[self.audio_pred])
        A_enc_weights = F.softmax(A_enc_weights, dim=-1)
        ex_audio = ex_audio * A_enc_weights.unsqueeze(2)
        ex_audio = F.normalize(torch.sum(ex_audio, dim=1), dim=-1)
        return ex_audio
    # image expert + IB[-2] + EVA_18B[-1]
    def proj_image(self, x:List[torch.Tensor]):
        with torch.no_grad():
            unieva_image = torch.mean(torch.stack([self.ib_projs[i].proj_image(x[-2]) for i in range(7)], dim=1), dim=1)*0.1
            unieva_image = F.normalize(unieva_image + x[-1]*0.9, dim=-1)
            
            ex_image = [self.projs[i + self.A_proj_nums].proj_image(x[i//7]) for i in range(self.V_proj_nums)] + [unieva_image]
            ex_image = torch.stack(ex_image, dim=1)
            ex_image = seg_mean(ex_image, 7, dim=1)
            B = len(x[-1])
        
        V_enc_weights = self.V_enc_router(x[self.image_pred])
        V_enc_weights = F.softmax(V_enc_weights, dim=-1)
        # V_proj_weights = torch.cat([torch.ones([B, 7], device=ex_image.device) *1/7 * V_enc_weights[:,i:i+1] for i in range(self.V_enc_nums - 1)] + [V_enc_weights[:,-1:]], dim=1)
        ex_image = ex_image * V_enc_weights.unsqueeze(2)
        ex_image = F.normalize(torch.sum(ex_image, dim=1), dim=-1)
        return ex_image
    # point expert
    def proj_point(self, x:List[torch.Tensor]):
        with torch.no_grad():
            ex_point = [self.projs[i + self.A_proj_nums + self.V_proj_nums - self.P_proj_nums].proj_point(x[i//7]) for i in range(self.P_proj_nums)]
            ex_point = torch.stack(ex_point, dim=1)
            ex_point = seg_mean(ex_point, 7, dim=1)
            B = len(x[-1])
            
        P_enc_weights = self.P_enc_router(x[self.point_pred])
        P_enc_weights = F.softmax(P_enc_weights, dim=-1)
        # P_proj_weights = torch.cat([torch.ones([B, 7], device=ex_point.device) *1/7 * P_enc_weights[:,i:i+1] for i in range(self.P_enc_nums)], dim=1)
        ex_point = ex_point * P_enc_weights.unsqueeze(2)
        ex_point = F.normalize(torch.sum(ex_point, dim=1), dim=-1)
        return ex_point
    # text expert[AT VT PVT] + IB[-2] + EVA_18B[-1]
    def proj_text(self, x:List[torch.Tensor]):
        
        with torch.no_grad():
            unieva_text = torch.mean(torch.stack([self.ib_projs[i].proj_text(x[-2]) for i in range(7)], dim=1), dim=1)*0.1
            unieva_text = F.normalize(unieva_text + x[-1]*0.9, dim=-1)
            
            ex_text = [self.projs[i].proj_text(x[i//7]) for i in range(self.T_proj_nums)] + [unieva_text]
            ex_text = torch.stack(ex_text, dim=1)
            ex_text = seg_mean(ex_text, 7, dim=1)
            B = len(x[-1])
        # x_pred = [x[idx_p] for idx_p in self.text_pred]
        # T_enc_weights = self.T_enc_router(torch.cat(x_pred, dim=1))
        T_enc_weights = self.get_text_enc_pred(x)
        T_enc_weights = F.softmax(T_enc_weights*10, dim=-1)
        
        # T_proj_weights = torch.cat([torch.ones([B, 7], device=ex_text.device) *1/7 * T_enc_weights[:,i:i+1] for i in range(self.T_enc_nums - 1)] + [T_enc_weights[:,-1:]], dim=1)
        # ex_text = ex_text * F.normalize(T_proj_weights, p=1, dim=-1).unsqueeze(2)
        ex_text = ex_text * T_enc_weights.unsqueeze(2)
        ex_text = F.normalize(torch.sum(ex_text, dim=1), dim=-1)
        return ex_text

    def get_text_enc_pred(self, x:List[torch.Tensor]):
        x_pred = [x[idx_p] for idx_p in self.text_pred]
        T_enc_weights = self.T_enc_router(torch.cat(x_pred, dim=1))
        return T_enc_weights
        
    def get_device(self):
        return self.projs[0].get_device()

class ATVP_router_wo18B(torch.nn.Module):
    def __init__(self, 
                 weights, 
                 config,
                 pretrained=True):
        super(ATVP_router_wo18B, self).__init__()
        # router = MOE_Router(in_dim=768, hidden_dim=100, out_dim=sum(projs)+1, bias=False, activation='tanh')
        # print(projs)
        self.A_proj_nums = config['A_num'] * 7
        self.V_proj_nums = (config['V_num']+config['P_num']) * 7
        self.P_proj_nums = config['P_num'] * 7
        self.T_proj_nums = (config['A_num']+config['V_num']+config['P_num']) * 7
        
        self.A_enc_nums  = config['A_num']+1
        self.V_enc_nums  = config['V_num']+config['P_num']+1
        self.P_enc_nums  = config['P_num']
        self.T_enc_nums  = (config['A_num']+config['V_num']+config['P_num'])+1
        
        print(self.A_proj_nums, self.V_proj_nums, self.P_proj_nums, self.T_proj_nums)
        print(self.A_enc_nums , self.V_enc_nums , self.P_enc_nums , self.T_enc_nums)
        
        self.A_enc_router = MOE_Router_Sig(in_dim=config['A_pred_dim'], out_dim=self.A_enc_nums)
        self.V_enc_router = MOE_Router_Sig(in_dim=config['V_pred_dim'], out_dim=self.V_enc_nums)
        self.P_enc_router = MOE_Router_Sig(in_dim=config['P_pred_dim'], out_dim=self.P_enc_nums)
        self.T_enc_router = MOE_Router_Sig(in_dim=config['T_pred_dim'], out_dim=self.T_enc_nums)
        
        self.projs = []
        for i in range(config['A_num']):
            self.projs += [Reaction_Head_mini(in_dim=config['in_dim']['AT'][i], out_dim=1536) for _ in range(7)]
        for i in range(config['V_num']):
            self.projs += [Reaction_Head_mini(in_dim=config['in_dim']['VT'][i], out_dim=1536) for _ in range(7)]
        for i in range(config['P_num']):
            self.projs += [Reaction_Head_mini(in_dim=config['in_dim']['PVT'][i], out_dim=1536) for _ in range(7)]
        self.projs = nn.ModuleList(self.projs)
        
        self.ib_projs = nn.ModuleList([Reaction_Head_mini(in_dim=1024, out_dim=1536) for _ in range(7)])
        # ib_weights = get_ib_paths()
        
        if pretrained:
            self.load_state_dict(torch.load(weights, map_location='cpu'), strict=False)
            
        self.audio_pred = config['A_pred']
        self.image_pred = config['V_pred']
        self.point_pred = config['P_pred']
        self.text_pred  = config['T_pred']
        
    # audio expert + IB[-1]
    def proj_audio(self, x:[torch.Tensor]):
        device = self.get_device()
        with torch.no_grad():
            x = [x_i.to(device) for x_i in x]
            ex_audio = [self.projs[i].proj_audio(x[i//7]) for i in range(self.A_proj_nums)] + [self.ib_projs[i].proj_audio(x[-1]) for i in range(7)]
            ex_audio = torch.stack(ex_audio, dim=1)
            ex_audio = seg_mean(ex_audio, 7, dim=1)
            B = len(x[-1])
            # print(ex_audio.shape, self.A_weights.shape)
        A_enc_weights = self.A_enc_router(x[self.audio_pred])
        A_enc_weights = F.softmax(A_enc_weights, dim=-1)
        ex_audio = ex_audio * A_enc_weights.unsqueeze(2)
        ex_audio = F.normalize(torch.sum(ex_audio, dim=1), dim=-1)
        return ex_audio
    # image expert + IB[-1]
    def proj_image(self, x:[torch.Tensor]):
        device = self.get_device()
        with torch.no_grad():
            x = [x_i.to(device) for x_i in x]
            unieva_image = F.normalize(torch.mean(torch.stack([self.ib_projs[i].proj_image(x[-1]) for i in range(7)], dim=1), dim=1), dim=-1)
            ex_image = [self.projs[i + self.A_proj_nums].proj_image(x[i//7]) for i in range(self.V_proj_nums)] + [unieva_image]
            ex_image = torch.stack(ex_image, dim=1)
            ex_image = seg_mean(ex_image, 7, dim=1)
            B = len(x[-1])
        
        V_enc_weights = self.V_enc_router(x[self.image_pred])
        V_enc_weights = F.softmax(V_enc_weights, dim=-1)
        # V_proj_weights = torch.cat([torch.ones([B, 7], device=ex_image.device) *1/7 * V_enc_weights[:,i:i+1] for i in range(self.V_enc_nums - 1)] + [V_enc_weights[:,-1:]], dim=1)
        ex_image = ex_image * V_enc_weights.unsqueeze(2)
        ex_image = F.normalize(torch.sum(ex_image, dim=1), dim=-1)
        return ex_image
    # point expert
    def proj_point(self, x:[torch.Tensor]):
        device = self.get_device()
        with torch.no_grad():
            x = [x_i.to(device) for x_i in x]
            ex_point = [self.projs[i + self.A_proj_nums + self.V_proj_nums - self.P_proj_nums].proj_point(x[i//7]) for i in range(self.P_proj_nums)]
            ex_point = torch.stack(ex_point, dim=1)
            ex_point = seg_mean(ex_point, 7, dim=1)
            B = len(x[-1])
            
        P_enc_weights = self.P_enc_router(x[self.point_pred])
        P_enc_weights = F.softmax(P_enc_weights, dim=-1)
        # P_proj_weights = torch.cat([torch.ones([B, 7], device=ex_point.device) *1/7 * P_enc_weights[:,i:i+1] for i in range(self.P_enc_nums)], dim=1)
        ex_point = ex_point * P_enc_weights.unsqueeze(2)
        ex_point = F.normalize(torch.sum(ex_point, dim=1), dim=-1)
        return ex_point
    # text expert[AT VT PVT] + IB[-1]
    def proj_text(self, x:[torch.Tensor]):
        device = self.get_device()
        with torch.no_grad():
            x = [x_i.to(device) for x_i in x]
            unieva_text = F.normalize(torch.mean(torch.stack([self.ib_projs[i].proj_text(x[-1]) for i in range(7)], dim=1), dim=1), dim=-1)
            
            ex_text = [self.projs[i].proj_text(x[i//7]) for i in range(self.T_proj_nums)] + [unieva_text]
            ex_text = torch.stack(ex_text, dim=1)
            ex_text = seg_mean(ex_text, 7, dim=1)
            B = len(x[-1])
        # x_pred = [x[idx_p] for idx_p in self.text_pred]
        # T_enc_weights = self.T_enc_router(torch.cat(x_pred, dim=1))
        T_enc_weights = self.get_text_enc_pred(x)
        T_enc_weights = F.softmax(T_enc_weights*10, dim=-1)
        
        # T_proj_weights = torch.cat([torch.ones([B, 7], device=ex_text.device) *1/7 * T_enc_weights[:,i:i+1] for i in range(self.T_enc_nums - 1)] + [T_enc_weights[:,-1:]], dim=1)
        # ex_text = ex_text * F.normalize(T_proj_weights, p=1, dim=-1).unsqueeze(2)
        ex_text = ex_text * T_enc_weights.unsqueeze(2)
        ex_text = F.normalize(torch.sum(ex_text, dim=1), dim=-1)
        return ex_text

    def get_text_enc_pred(self, x:[torch.Tensor]):
        x_pred = [x[idx_p] for idx_p in self.text_pred]
        T_enc_weights = self.T_enc_router(torch.cat(x_pred, dim=1))
        return T_enc_weights

    def get_device(self):
        return self.projs[0].get_device()
