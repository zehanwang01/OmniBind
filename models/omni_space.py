import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Function
import copy
from tqdm import tqdm
from projector import *
from utils import *
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
        config['V_pred'] = 4
        config['P_pred'] = 0
        config['T_pred'] = [2,9,10]
        config['A_pred_dim'] = 1024
        config['V_pred_dim'] = 1152
        config['P_pred_dim'] = 1024
        config['T_pred_dim'] = 1024 + 1152 + 1024
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
                 projs  = {'AT':[7, 7, 7, 7, 7], 'VT': [7, 7, 7, 7, 7], 'PVT': [7, 7, 7, 7]},
                 in_dim = {'AT':[512, 512, 1024, 1024, 1024], 'VT': [1024, 1024, 1280, 1024, 1152], 'PVT': [1024, 1024, 1024, 1024]},
                 out_dim= 1536,
                 config = get_config('Full'),
                 mode   = 'eval'):
        super(ATVP_router_UNI, self).__init__()

        self.A_proj_nums = sum(projs['AT'])
        self.V_proj_nums = sum(projs['VT'])
        self.P_proj_nums = sum(projs['PVT'])
        self.T_proj_nums = sum(projs['AT'])+sum(projs['VT'])+sum(projs['PVT'])
        
        self.A_enc_nums  = len(projs['AT'])+1
        self.V_enc_nums  = len(projs['VT'])+1
        self.P_enc_nums  = len(projs['PVT'])
        self.T_enc_nums  = len(projs['AT'])+len(projs['VT'])+len(projs['PVT'])+1
        
        print(self.A_proj_nums, self.V_proj_nums, self.P_proj_nums, self.T_proj_nums)
        print(self.A_enc_nums , self.V_enc_nums , self.P_enc_nums , self.T_enc_nums)
        
        self.A_enc_router = MOE_Router_Sig(in_dim=config['A_pred_dim'], out_dim=self.A_enc_nums)
        self.V_enc_router = MOE_Router_Sig(in_dim=config['V_pred_dim'], out_dim=self.V_enc_nums)
        self.P_enc_router = MOE_Router_Sig(in_dim=config['P_pred_dim'], out_dim=self.P_enc_nums)
        self.T_enc_router = MOE_Router_Sig(in_dim=config['T_pred_dim'], out_dim=self.T_enc_nums)
        
        self.projs = []
        for i in range(len(projs['AT'])):
            self.projs += [Reaction_Head_mini(in_dim=in_dim['AT'][i], out_dim=out_dim) for _ in range(projs['AT'][i])]
        for i in range(len(projs['VT'])):
            self.projs += [Reaction_Head_mini(in_dim=in_dim['VT'][i], out_dim=out_dim) for _ in range(projs['VT'][i])]
        for i in range(len(projs['PVT'])):
            self.projs += [Reaction_Head_mini(in_dim=in_dim['PVT'][i], out_dim=out_dim) for _ in range(projs['PVT'][i])]
        self.projs = nn.ModuleList(self.projs)
        
        self.ib_projs = nn.ModuleList([Reaction_Head_mini(in_dim=1024, out_dim=out_dim) for _ in range(7)])
        
        if mode == 'train':
            for i in tqdm(range(len(self.projs))):
                self.projs[i].load_state_dict(torch.load(weights[i], map_location='cpu'))
                self.projs[i].eval()
                self.projs[i].disable_grad()
        elif mode == 'eval':
            self.load_state_dict(torch.load(weights, map_location='cpu'), strict=False)
            
        self.audio_pred = config['A_pred']
        self.image_pred = config['V_pred']
        self.point_pred = config['P_pred']
        self.text_pred  = config['T_pred']
        
    # audio expert + IB[-1]
    def proj_audio(self, x:[torch.Tensor]):
        with torch.no_grad():
            ex_audio = [self.projs[i].proj_audio(x[i//7]) for i in range(self.A_proj_nums)] + [self.ib_projs[i].proj_audio(x[-1]) for i in range(7)]
            ex_audio = torch.stack(ex_audio, dim=1)
            seg_mean
            B = len(x[-1])
            # print(ex_audio.shape, self.A_weights.shape)
        A_enc_weights = self.A_enc_router(x[self.audio_pred])
        # A_enc_weights = F.softmax(A_enc_weights, dim=-1)
        
        if self.linear_norm:
            A_enc_weights = F.normalize(A_enc_weights, p=1, dim=-1)
        else:
            A_enc_weights = F.softmax(A_enc_weights, dim=-1)
        if self.domain_mean:
            A_proj_weights = torch.cat([torch.ones([B, 7], device=ex_audio.device) *1/35 * A_enc_weights[:,i:i+1] for i in range(self.A_enc_nums - 1)] + [A_enc_weights[:,-1:]], dim=1)
        else:        
            A_proj_weights = torch.cat([torch.ones([B, 7], device=ex_audio.device) *1/7 * A_enc_weights[:,i:i+1] for i in range(self.A_enc_nums - 1)] + [A_enc_weights[:,-1:]], dim=1)
        # print(A_proj_weights.shape)
        if self.fix_mean or self.loss_mode=='B':
            A_proj_weights = torch.cat([torch.ones([B, 7], device=ex_audio.device) *1/7  for i in range(self.A_enc_nums - 1)] + [torch.ones([B, 1], device=ex_audio.device)], dim=1)
        if self.fix_mean and self.domain_mean:
            A_proj_weights = torch.cat([torch.ones([B, 7], device=ex_audio.device) *1/7  for i in range(self.A_enc_nums - 1)] + [torch.ones([B, 1], device=ex_audio.device)], dim=1)
        
            
        ex_audio = ex_audio * F.normalize(A_proj_weights, p=1, dim=-1).unsqueeze(2)
        ex_audio = F.normalize(torch.sum(ex_audio, dim=1), dim=-1)
        return ex_audio
    # image expert + IB[-2] + EVA_18B[-1]
    def proj_image(self, x:[torch.Tensor]):
        with torch.no_grad():
            ex_image = [self.projs[i + self.A_proj_nums].proj_image(x[i//7]) for i in range(self.V_proj_nums)] + [x[-1]]
            ex_image = torch.stack(ex_image, dim=1)
            B = len(x[-1])
            # print(ex_image.shape, self.V_weights.shape)
        # V_enc_weights = self.V_enc_router(x[-1]) * torch.tensor([1,1,0,1,1,1]).unsqueeze(0).to(ex_image.device)
        V_enc_weights = self.V_enc_router(x[self.image_pred])
        # V_enc_weights = F.softmax(V_enc_weights, dim=-1)
        if self.linear_norm:
            V_enc_weights = F.normalize(V_enc_weights, p=1, dim=-1)
        else:
            V_enc_weights = F.softmax(V_enc_weights, dim=-1)
        
        if self.domain_mean:
            V_proj_weights = torch.cat([torch.ones([B, 7], device=ex_image.device) *1/28 * V_enc_weights[:,i:i+1] for i in range(0,4)] + 
                                       [torch.ones([B, 7], device=ex_image.device) *1/21 * V_enc_weights[:,i:i+1] for i in range(4,7)] + 
                                       [V_enc_weights[:,-1:]], dim=1)
        else:
            V_proj_weights = torch.cat([torch.ones([B, 7], device=ex_image.device) *1/7 * V_enc_weights[:,i:i+1] for i in range(self.V_enc_nums - 1)] + [V_enc_weights[:,-1:]], dim=1)
        # print(V_proj_weights.shape)
        
        if self.fix_mean or self.loss_mode=='B':
            V_proj_weights = torch.cat([torch.ones([B, 7], device=ex_image.device) *1/7  for i in range(self.V_enc_nums - 1)] + [torch.ones([B, 1], device=ex_image.device)], dim=1)
        if self.fix_mean and self.domain_mean:
            V_proj_weights = torch.cat([torch.ones([B, 7], device=ex_image.device) *1  for i in range(0,4)] + 
                                       [torch.ones([B, 7], device=ex_image.device) *1/14  for i in range(4,7)] + 
                                       [torch.ones([B, 1], device=ex_image.device) *1], dim=1)
        
        ex_image = ex_image * F.normalize(V_proj_weights, p=1, dim=-1).unsqueeze(2)
        ex_image = F.normalize(torch.sum(ex_image, dim=1), dim=-1)
        return ex_image
    # point expert
    def proj_point(self, x:[torch.Tensor]):
        with torch.no_grad():
            ex_point = [self.projs[i + self.A_proj_nums + self.V_proj_nums].proj_point(x[i//7]) for i in range(self.P_proj_nums)]
            ex_point = torch.stack(ex_point, dim=1)
            B = len(x[-1])
            
        P_enc_weights = self.P_enc_router(x[self.point_pred])
        if self.linear_norm:
            P_enc_weights = F.normalize(P_enc_weights, p=1, dim=-1)
        else:
            P_enc_weights = F.softmax(P_enc_weights, dim=-1)
        P_proj_weights = torch.cat([torch.ones([B, 7], device=ex_point.device) *1/7 * P_enc_weights[:,i:i+1] for i in range(self.P_enc_nums)], dim=1)
        
        if self.fix_mean or self.loss_mode=='B':
            P_proj_weights = torch.cat([torch.ones([B, 7], device=ex_point.device) *1/7  for i in range(self.P_enc_nums)], dim=1)
        
        ex_point = ex_point * F.normalize(P_proj_weights, p=1, dim=-1).unsqueeze(2)
        ex_point = F.normalize(torch.sum(ex_point, dim=1), dim=-1)
        return ex_point
    # text expert[AT VT PVT] + IB[-2] + EVA_18B[-1]
    def proj_text(self, x:[torch.Tensor]):
        
        with torch.no_grad():
            ex_text = [self.projs[i].proj_text(x[i//7]) for i in range(self.T_proj_nums)] + [x[-1]]
            ex_text = torch.stack(ex_text, dim=1)
            B = len(x[-1])
        # x_pred = [x[idx_p] for idx_p in self.text_pred]
        # T_enc_weights = self.T_enc_router(torch.cat(x_pred, dim=1))
        if self.gate:
            T_enc_weights = self.get_text_enc_pred(x) * self.get_text_gate_pred(x)
        else:
            T_enc_weights = self.get_text_enc_pred(x)
            # T_enc_weights = self.get_text_enc_pred(x) * torch.cat([torch.zeros(5), torch.ones(4), torch.zeros(3), torch.ones(1)*60]).float().unsqueeze(0).to(x[0].device)
            # T_enc_weights = torch.cat([torch.zeros(B,3), torch.zeros(B,3), torch.ones(B,3), torch.zeros(B,1)], dim=1).float().to(ex_text.device)
        if self.linear_norm:
            T_enc_weights = F.normalize(T_enc_weights, p=1, dim=-1)
        else:
            T_enc_weights = F.softmax(T_enc_weights*10, dim=-1)
        if self.domain_mean:
            T_proj_weights = torch.cat([torch.ones([B, 7], device=ex_text.device) *1/35 * T_enc_weights[:,i:i+1] for i in range(0,5)] + 
                                       [torch.ones([B, 7], device=ex_text.device) *1/28 * T_enc_weights[:,i:i+1] for i in range(5,9)] + 
                                       [torch.ones([B, 7], device=ex_text.device) *1/21 * T_enc_weights[:,i:i+1] for i in range(9,12)] + 
                                       [T_enc_weights[:,-1:]], dim=1)
        else:
            T_proj_weights = torch.cat([torch.ones([B, 7], device=ex_text.device) *1/7 * T_enc_weights[:,i:i+1] for i in range(self.T_enc_nums - 1)] + [T_enc_weights[:,-1:]], dim=1)
        if self.fix_mean:
            T_proj_weights = torch.cat([torch.ones([B, 7], device=ex_text.device) *1/7  for i in range(self.T_enc_nums - 1)] + [torch.ones([B, 1], device=ex_text.device)], dim=1)
        if self.fix_mean and self.domain_mean:
            T_proj_weights = torch.cat([torch.ones([B, 7], device=ex_text.device) *1/21 for i in range(0,5)] + 
                                       [torch.ones([B, 7], device=ex_text.device) *1/3 for i in range(5,9)] + 
                                       [torch.ones([B, 7], device=ex_text.device) *1/28  for i in range(9,12)] + 
                                       [torch.ones([B, 1], device=ex_text.device) *1], dim=1)
        # print(ex_text.shape, T_proj_weights.shape)
        ex_text = ex_text * F.normalize(T_proj_weights, p=1, dim=-1).unsqueeze(2)
        # ex_text = ex_text * F.softmax(T_proj_weights, dim=-1).unsqueeze(2)
        ex_text = F.normalize(torch.sum(ex_text, dim=1), dim=-1)
        return ex_text

    def get_text_enc_pred(self, x:[torch.Tensor]):
        x_pred = [x[idx_p] for idx_p in self.text_pred]
        T_enc_weights = self.T_enc_router(torch.cat(x_pred, dim=1))
        return T_enc_weights
    def get_audio_enc_pred(self, x:[torch.Tensor]):
        A_enc_weights = self.A_enc_router(x[self.audio_pred])
        return A_enc_weights
    def get_image_enc_pred(self, x:[torch.Tensor]):
        V_enc_weights = self.V_enc_router(x[self.image_pred])
        return V_enc_weights

    def get_text_gate_pred(self, x:[torch.Tensor]):
        x_pred = [x[idx_p] for idx_p in self.text_pred]
        T_gate_pred = self.T_gate(torch.cat(x_pred, dim=1))
        return T_gate_pred
    
    def get_device(self):
        return self.projs[0].get_device()

class ATVP_router_wo18B(torch.nn.Module):
    def __init__(self, 
                 weights, 
                 projs  = {'AT':[7, 7, 7, 7, 7], 'VT': [7, 7, 7, 7, 7], 'PVT': [7, 7, 7, 7], 'AVT': [7]},
                 in_dim = {'AT':[512, 512, 1024, 1024, 1024], 'VT': [1024, 1024, 1280, 1024, 1152], 'PVT': [1024, 1024, 1024, 1024], 'AVT': [1024]},
                 out_dim=1536,
                 config = get_config('Large'),
                 mode = 'train'):
        super(ATVP_router_wo18B, self).__init__()
        # router = MOE_Router(in_dim=768, hidden_dim=100, out_dim=sum(projs)+1, bias=False, activation='tanh')
        self.linear_norm = False
        # print(projs)
        self.A_proj_nums = sum(projs['AT'])
        self.V_proj_nums = sum(projs['VT'])
        self.P_proj_nums = sum(projs['PVT'])
        self.T_proj_nums = sum(projs['AT'])+sum(projs['VT'])+sum(projs['PVT'])
        
        self.A_enc_nums  = len(projs['AT'])+1
        self.V_enc_nums  = len(projs['VT'])+len(projs['PVT'])+1
        self.P_enc_nums  = len(projs['PVT'])
        self.T_enc_nums  = len(projs['AT'])+len(projs['VT'])+len(projs['PVT'])+1
        
        print(self.A_proj_nums, self.V_proj_nums, self.P_proj_nums, self.T_proj_nums)
        print(self.A_enc_nums , self.V_enc_nums , self.P_enc_nums , self.T_enc_nums)
        
        self.A_enc_router = MOE_Router_Sig(in_dim=config['A_pred_dim'], out_dim=self.A_enc_nums)
        self.V_enc_router = MOE_Router_Sig(in_dim=config['V_pred_dim'], out_dim=self.V_enc_nums)
        self.P_enc_router = MOE_Router_Sig(in_dim=config['P_pred_dim'], out_dim=self.P_enc_nums)
        self.T_enc_router = MOE_Router_Sig(in_dim=config['T_pred_dim'], out_dim=self.T_enc_nums)
        
        self.projs = []
        for i in range(len(projs['AT'])):
            self.projs += [Reaction_Head_mini(in_dim=in_dim['AT'][i], out_dim=out_dim) for _ in range(projs['AT'][i])]
        for i in range(len(projs['VT'])):
            self.projs += [Reaction_Head_mini(in_dim=in_dim['VT'][i], out_dim=out_dim) for _ in range(projs['VT'][i])]
        for i in range(len(projs['PVT'])):
            self.projs += [Reaction_Head_mini(in_dim=in_dim['PVT'][i], out_dim=out_dim) for _ in range(projs['PVT'][i])]
        self.projs = nn.ModuleList(self.projs)
        
        self.ib_projs = nn.ModuleList([Reaction_Head_mini(in_dim=1024, out_dim=out_dim) for _ in range(7)])
        ib_weights = get_ib_paths()
        
        if mode == 'train':
            for i in tqdm(range(len(self.projs))):
                self.projs[i].load_state_dict(torch.load(weights[i], map_location='cpu'))
                self.projs[i].eval()
                self.projs[i].disable_grad()
            for i in tqdm(range(len(self.ib_projs))):
                self.ib_projs[i].load_state_dict(torch.load(ib_weights[i], map_location='cpu'))
                self.ib_projs[i].eval()
                self.ib_projs[i].disable_grad()
        elif mode == 'eval':
            self.load_state_dict(torch.load(weights, map_location='cpu'), strict=False)
            
        self.audio_pred = config['A_pred']
        self.image_pred = config['V_pred']
        self.point_pred = config['P_pred']
        self.text_pred  = config['T_pred']
        
    def proj_audio(self, x:[torch.Tensor]):
        with torch.no_grad():
            ex_audio = [self.projs[i].proj_audio(x[i//7]) for i in range(self.A_proj_nums)] + [self.ib_projs[i].proj_audio(x[-1]) for i in range(7)]
            ex_audio = torch.stack(ex_audio, dim=1)
            B = len(x[-1])
            # print(ex_audio.shape, self.A_weights.shape)
        A_enc_weights = self.A_enc_router(x[self.audio_pred])
        # A_enc_weights = F.softmax(A_enc_weights, dim=-1)
        if self.linear_norm:
            A_enc_weights = F.normalize(A_enc_weights, p=1, dim=-1)
        else:
            A_enc_weights = F.softmax(A_enc_weights, dim=-1)
        if self.domain_mean:
            A_proj_weights = torch.cat([torch.ones([B, 7], device=ex_audio.device) *1/35 * A_enc_weights[:,i:i+1] for i in range(self.A_enc_nums - 1)] + 
                                       [torch.ones([B, 7], device=ex_audio.device) *1/7 * A_enc_weights[:,-1:]], dim=1)
        else:        
            A_proj_weights = torch.cat([torch.ones([B, 7], device=ex_audio.device) *1/7 * A_enc_weights[:,i:i+1] for i in range(self.A_enc_nums)], dim=1)
        # print(A_proj_weights.shape)
        if self.fix_mean:
            A_proj_weights = torch.cat([torch.ones([B, 7], device=ex_audio.device) *1/7  for i in range(self.A_enc_nums)], dim=1)
        
        ex_audio = ex_audio * F.normalize(A_proj_weights, p=1, dim=-1).unsqueeze(2)
        ex_audio = F.normalize(torch.sum(ex_audio, dim=1), dim=-1)
        return ex_audio
    
    def proj_image(self, x:[torch.Tensor]):
        with torch.no_grad():
            ex_image = [self.projs[i + self.A_proj_nums].proj_image(x[i//7]) for i in range(self.V_proj_nums + self.P_proj_nums)] + [self.ib_projs[i].proj_image(x[-1]) for i in range(7)]
            ex_image = torch.stack(ex_image, dim=1)
            B = len(x[-1])
            # print(ex_image.shape, self.V_weights.shape)
        # V_enc_weights = self.V_enc_router(x[-1]) * torch.tensor([1,1,0,1,1,1]).unsqueeze(0).to(ex_image.device)
        V_enc_weights = self.V_enc_router(x[self.image_pred])
        # V_enc_weights = F.softmax(V_enc_weights, dim=-1)
        if self.linear_norm:
            V_enc_weights = F.normalize(V_enc_weights, p=1, dim=-1)
        else:
            V_enc_weights = F.softmax(V_enc_weights, dim=-1)
        
        if self.domain_mean:
            # V_proj_weights = torch.cat([torch.ones([B, 7], device=ex_image.device) *1/28 * V_enc_weights[:,i:i+1] for i in range(0,4)] + 
            #                            [torch.ones([B, 7], device=ex_image.device) *1/21 * V_enc_weights[:,i:i+1] for i in range(4,7)] + 
            #                         #    [V_enc_weights[:,-1:]], dim=1)
            #                            [torch.ones([B, 7], device=ex_image.device) *1/7 * V_enc_weights[:,-1:]], dim=1)
            V_proj_weights = torch.cat([torch.ones([B, 7], device=ex_image.device) *1/7 * V_enc_weights[:,i:i+1] for i in range(0,3)], dim=1)
        else:
            V_proj_weights = torch.cat([torch.ones([B, 7], device=ex_image.device) *1/7 * V_enc_weights[:,i:i+1] for i in range(self.V_enc_nums)], dim=1)
        # print(V_proj_weights.shape)
        
        if self.fix_mean:
            V_proj_weights = torch.cat([torch.ones([B, 7], device=ex_image.device) *1/7  for i in range(self.V_enc_nums)], dim=1)
        
        ex_image = ex_image * F.normalize(V_proj_weights, p=1, dim=-1).unsqueeze(2)
        ex_image = F.normalize(torch.sum(ex_image, dim=1), dim=-1)
        return ex_image
    
    def proj_point(self, x:[torch.Tensor]):
        with torch.no_grad():
            ex_point = [self.projs[i + self.A_proj_nums + self.V_proj_nums].proj_point(x[i//7]) for i in range(self.P_proj_nums)]
            ex_point = torch.stack(ex_point, dim=1)
            B = len(x[-1])
            
        P_enc_weights = self.P_enc_router(x[self.point_pred])
        if self.linear_norm:
            P_enc_weights = F.normalize(P_enc_weights, p=1, dim=-1)
        else:
            P_enc_weights = F.softmax(P_enc_weights, dim=-1)
        P_proj_weights = torch.cat([torch.ones([B, 7], device=ex_point.device) *1/7 * P_enc_weights[:,i:i+1] for i in range(self.P_enc_nums)], dim=1)
        
        if self.fix_mean:
            P_proj_weights = torch.cat([torch.ones([B, 7], device=ex_point.device) *1/7  for i in range(self.P_enc_nums)], dim=1)
        
        ex_point = ex_point * F.normalize(P_proj_weights, p=1, dim=-1).unsqueeze(2)
        ex_point = F.normalize(torch.sum(ex_point, dim=1), dim=-1)
        return ex_point
    
    def proj_text(self, x:[torch.Tensor]):
        
        with torch.no_grad():
            ex_text = [self.projs[i].proj_text(x[i//7]) for i in range(self.T_proj_nums)] + [self.ib_projs[i].proj_text(x[-1]) for i in range(7)]
            ex_text = torch.stack(ex_text, dim=1)
            B = len(x[-1])
        # x_pred = [x[idx_p] for idx_p in self.text_pred]
        # T_enc_weights = self.T_enc_router(torch.cat(x_pred, dim=1))
        if self.gate:
            T_enc_weights = self.get_text_enc_pred(x) * self.get_text_gate_pred(x)
        else:
            T_enc_weights = self.get_text_enc_pred(x)
            # T_enc_weights = torch.cat([torch.zeros(B,3), torch.zeros(B,3), torch.ones(B,3), torch.zeros(B,1)], dim=1).float().to(ex_text.device)
        if self.linear_norm:
            T_enc_weights = F.normalize(T_enc_weights, p=1, dim=-1)
        else:
            T_enc_weights = F.softmax(T_enc_weights, dim=-1)
        if self.domain_mean:
            # T_proj_weights = torch.cat([torch.ones([B, 7], device=ex_text.device) *1/35 * T_enc_weights[:,i:i+1] for i in range(0,5)] + 
            #                            [torch.ones([B, 7], device=ex_text.device) *1/28 * T_enc_weights[:,i:i+1] for i in range(5,9)] + 
            #                            [torch.ones([B, 7], device=ex_text.device) *1/21 * T_enc_weights[:,i:i+1] for i in range(9,12)] + 
            #                            [torch.ones([B, 7], device=ex_text.device) *1/7 * T_enc_weights[:,-1:]], dim=1)
            T_proj_weights = torch.cat([torch.ones([B, 7], device=ex_text.device) *1/35 * T_enc_weights[:,i:i+1] for i in range(0,5)] + 
                                       [torch.ones([B, 7], device=ex_text.device) *1/7 * T_enc_weights[:,i:i+1] for i in range(5,6)] + 
                                       [torch.ones([B, 7], device=ex_text.device) *1/7 * T_enc_weights[:,i:i+1] for i in range(6,7)] + 
                                       [torch.ones([B, 7], device=ex_text.device) *1/7 * T_enc_weights[:,-1:]], dim=1)
        else:
            T_proj_weights = torch.cat([torch.ones([B, 7], device=ex_text.device) *1/7 * T_enc_weights[:,i:i+1] for i in range(self.T_enc_nums)], dim=1)
        if self.fix_mean:
            T_proj_weights = torch.cat([torch.ones([B, 7], device=ex_text.device) *1/7  for i in range(self.T_enc_nums)], dim=1)
        
        # print(ex_text.shape, T_proj_weights.shape)
        ex_text = ex_text * F.normalize(T_proj_weights, p=1, dim=-1).unsqueeze(2)
        # ex_text = ex_text * F.softmax(T_proj_weights, dim=-1).unsqueeze(2)
        ex_text = F.normalize(torch.sum(ex_text, dim=1), dim=-1)
        return ex_text

    def get_text_enc_pred(self, x:[torch.Tensor]):
        x_pred = [x[idx_p] for idx_p in self.text_pred]
        T_enc_weights = self.T_enc_router(torch.cat(x_pred, dim=1))
        return T_enc_weights
    def get_audio_enc_pred(self, x:[torch.Tensor]):
        A_enc_weights = self.A_enc_router(x[self.audio_pred])
        return A_enc_weights
    def get_image_enc_pred(self, x:[torch.Tensor]):
        V_enc_weights = self.V_enc_router(x[self.image_pred])
        return V_enc_weights

    def get_text_gate_pred(self, x:[torch.Tensor]):
        x_pred = [x[idx_p] for idx_p in self.text_pred]
        T_gate_pred = self.T_gate(torch.cat(x_pred, dim=1))
        return T_gate_pred
    
    def get_device(self):
        return self.projs[0].get_device()
    
    def get_balance(self, txt):
        T_enc_weights = self.T_enc_router(txt)
        T_import = T_enc_weights.sum(0)
        T_bal = cv_squared(T_import)
        
        # V_enc_weights = self.V_enc_router(img)
        # V_import = V_enc_weights.sum(0)
        # V_bal = cv_squared(V_import)
        # return T_bal + V_bal
        return T_bal

class OmniBind_Space(torch.nn.Module):
    def __init__(self):
        super(OmniBind_Space, self).__init__()
        self.expert_pool = None
        self.atvp_router = None
        
    @torch.no_grad()
    def emb_audios(self, audio_files:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_audios\' method')
    @torch.no_grad()
    def emb_images(self, image_files:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_images\' method')
    @torch.no_grad()
    def emb_points(self, point_files:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_points\' method')
    @torch.no_grad()
    def emb_texts(self, texts:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_texts\' method')

# Base
class OmniBind_Base(OmniBind_Space):
    def __init__(self):
        super(OmniBind_Space, self).__init__()
        self.expert_pool = None
        self.atvp_router = None
        
    @torch.no_grad()
    def emb_audios(self, audio_files:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_audios\' method')
    @torch.no_grad()
    def emb_images(self, image_files:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_images\' method')
    @torch.no_grad()
    def emb_points(self, point_files:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_points\' method')
    @torch.no_grad()
    def emb_texts(self, texts:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_texts\' method')

# Large
class OmniBind_Large(OmniBind_Space):
    def __init__(self):
        super(OmniBind_Space, self).__init__()
        self.expert_pool = None
        self.atvp_router = None
        
    @torch.no_grad()
    def emb_audios(self, audio_files:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_audios\' method')
    @torch.no_grad()
    def emb_images(self, image_files:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_images\' method')
    @torch.no_grad()
    def emb_points(self, point_files:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_points\' method')
    @torch.no_grad()
    def emb_texts(self, texts:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_texts\' method')
    
# Full
class OmniBind_Full(OmniBind_Space):
    def __init__(self):
        super(OmniBind_Space, self).__init__()
        self.expert_pool = None
        self.atvp_router = None
        
    @torch.no_grad()
    def emb_audios(self, audio_files:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_audios\' method')
    @torch.no_grad()
    def emb_images(self, image_files:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_images\' method')
    @torch.no_grad()
    def emb_points(self, point_files:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_points\' method')
    @torch.no_grad()
    def emb_texts(self, texts:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_texts\' method')