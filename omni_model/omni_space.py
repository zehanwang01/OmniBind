import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Function
import copy
from tqdm import tqdm
from projector import *
from omni_utils import *
from experts import Expert_Pool
from router import *
from paths import *

class OmniBind_Space(torch.nn.Module):
    def __init__(self):
        super(OmniBind_Space, self).__init__()
        self.expert_pool = None
        self.atvp_router = None
        check_download()
        
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
    def __init__(self, pretrained=True):
        super(OmniBind_Base, self).__init__()
        self.config = get_config('Base')
        self.atvp_router = ATVP_router_wo18B(weights=OMNIBIND_BASE, config=self.config, pretrained=pretrained)
        self.expert_pool = Expert_Pool(config=self.config, pretrained=pretrained)
        
    @torch.no_grad()
    def emb_audios(self, audio_files:[str])->Tensor:
        audio_embs = self.expert_pool.emb_audios(audio_files=audio_files)
        audio_embs = self.atvp_router.proj_audio(audio_embs)
        return audio_embs
    @torch.no_grad()
    def emb_images(self, image_files:[str])->Tensor:
        image_embs = self.expert_pool.emb_images(image_files=image_files)
        image_embs = self.atvp_router.proj_image(image_embs)
        return image_embs
    @torch.no_grad()
    def emb_points(self, point_files:[str])->Tensor:
        point_embs = self.expert_pool.emb_points(point_files=point_files)
        point_embs = self.atvp_router.proj_point(point_embs)
        return point_embs
    @torch.no_grad()
    def emb_texts(self, texts:[str])->Tensor:
        text_embs = self.expert_pool.emb_texts(texts=texts)
        text_embs = self.atvp_router.proj_text(text_embs)
        return text_embs

# Large
class OmniBind_Large(OmniBind_Space):
    def __init__(self, pretrained=True):
        super(OmniBind_Large, self).__init__()
        self.config = get_config('Large')
        self.expert_pool = Expert_Pool(config=self.config, pretrained=pretrained)
        self.atvp_router = ATVP_router_wo18B(weights=OMNIBIND_LARGE, config=self.config, pretrained=pretrained)
        
    @torch.no_grad()
    def emb_audios(self, audio_files:[str])->Tensor:
        audio_embs = self.expert_pool.emb_audios(audio_files=audio_files)
        audio_embs = self.atvp_router.proj_audio(audio_embs)
        return audio_embs
    @torch.no_grad()
    def emb_images(self, image_files:[str])->Tensor:
        image_embs = self.expert_pool.emb_images(image_files=image_files)
        image_embs = self.atvp_router.proj_image(image_embs)
        return image_embs
    @torch.no_grad()
    def emb_points(self, point_files:[str])->Tensor:
        point_embs = self.expert_pool.emb_points(point_files=point_files)
        point_embs = self.atvp_router.proj_point(point_embs)
        return point_embs
    @torch.no_grad()
    def emb_texts(self, texts:[str])->Tensor:
        text_embs = self.expert_pool.emb_texts(texts=texts)
        text_embs = self.atvp_router.proj_text(text_embs)
        return text_embs
    
# Full
class OmniBind_Full(OmniBind_Space):
    def __init__(self, pretrained=True):
        super(OmniBind_Full, self).__init__()
        self.config = get_config('Full')
        self.expert_pool = Expert_Pool(config=self.config, pretrained=pretrained)
        self.atvp_router = ATVP_router_UNI(weights=OMNIBIND_FULL, config=self.config, pretrained=pretrained)
        
    @torch.no_grad()
    def emb_audios(self, audio_files:[str])->Tensor:
        audio_embs = self.expert_pool.emb_audios(audio_files=audio_files)
        audio_embs = self.atvp_router.proj_audio(audio_embs)
        return audio_embs
    @torch.no_grad()
    def emb_images(self, image_files:[str])->Tensor:
        image_embs = self.expert_pool.emb_images(image_files=image_files)
        image_embs = self.atvp_router.proj_image(image_embs)
        return image_embs
    @torch.no_grad()
    def emb_points(self, point_files:[str])->Tensor:
        point_embs = self.expert_pool.emb_points(point_files=point_files)
        point_embs = self.atvp_router.proj_point(point_embs)
        return point_embs
    @torch.no_grad()
    def emb_texts(self, texts:[str])->Tensor:
        text_embs = self.expert_pool.emb_texts(texts=texts)
        text_embs = self.atvp_router.proj_text(text_embs)
        return text_embs