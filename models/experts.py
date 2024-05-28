import torch
from torch import nn, Tensor
from torch.nn import functional as F
import laion_clap
from projector import Reaction_Head_mini
from paths import *
from imagebind_audio import imagebind_huge_audio

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor, AutoProcessor
from PIL import Image
import os

from wavcaps.ase_model import ASE
from ruamel import yaml
import librosa
from Uni3D.utils import utils as uni3d_utils
import Uni3D.models.uni3d as uni3d_models
import numpy as np

class Expert(nn.modules):
    def __init__(self):
        super(Expert, self).__init__()
        
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
# IB
class ImageBind(Expert):
    def __init__(self):
        super(ImageBind, self).__init__()
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        
    @torch.no_grad()
    def emb_audios(self, audio_files:[str])->Tensor:
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(audio_files, self.get_device()),
        }
        return F.normalize(self.model(inputs)[ModalityType.AUDIO])
    
    @torch.no_grad()
    def emb_images(self, image_files:[str])->Tensor:
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(image_files, self.get_device()),
        }
        return F.normalize(self.model(inputs)[ModalityType.VISION])
    
    @torch.no_grad()
    def emb_texts(self, texts:[str])->Tensor:
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(texts, self.get_device()),
        }
        return F.normalize(self.model(inputs)[ModalityType.TEXT])
    
    def get_device(self):
        return next(self.parameters()).device

# WavCaps*3
class WavCaps(Expert):
    def __init__(self, mode = 'PT'):
        super(WavCaps, self).__init__()
        with open("settings/inference.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.model = ASE(config)
        if mode == 'PT':
            cp_path = WAVCAPS_PT_PATH
        if mode == 'FT_aud':
            cp_path = WAVCAPS_FT_AUD_PATH
        if mode == 'FT_clo':
            cp_path = WAVCAPS_FT_CLO_PATH

        cp = torch.load(cp_path, map_location='cpu')
        self.model.load_state_dict(cp['model'])
        self.model.eval()
        
    @torch.no_grad()
    def emb_audios(self, audio_files:[str]):
        device = self.get_device()
        audio_embs = []
        for audio_path in audio_files:
            audio, _ = librosa.load(audio_path, sr=32000, mono=True)
            audio = torch.tensor(audio).unsqueeze(0).to(device)
            if audio.shape[-1] < 32000 * 10:
                pad_length = 32000 * 10 - audio.shape[-1]
                audio = F.pad(audio, [0, pad_length], "constant", 0.0)
            elif audio.shape[-1] > 32000 * 10:
                audio = audio[:, : 32000 * 10]
            # print(audio.shape)
            audio_emb = self.model.encode_audio(audio)
            audio_embs.append(audio_emb)
        audio_embs = torch.cat(audio_embs)
        audio_embs = F.normalize(audio_embs, dim=-1)
        return audio_embs
    
    @torch.no_grad()
    def emb_texts(self, texts:[str]):
        text_embs = []
        batch = 8
        split = (len(texts) - 1)// batch + 1
        with torch.no_grad():
            for i in range(split):
                text_embs.append(self.model.encode_text(texts[i*batch:(i+1)*batch]))
        text_embs = torch.cat(text_embs)
        text_embs = F.normalize(text_embs, dim=-1)
        return text_embs

    def get_device(self):
        return next(self.parameters()).device

# CLAP_G CLAP_M
class CLAP(Expert):
    def __init__(self, mode = 'G'):
        super(CLAP, self).__init__()
        if mode == 'G':
            self.clap = laion_clap.CLAP_Module(enable_fusion=True, device='cpu')
            self.clap.load_ckpt(CLAP_G_PATH)
        if mode == 'M':
            self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base', device='cpu')
            self.clap.load_ckpt(CLAP_M_PATH)
        
    def emb_audios(self, audio_files:[str]):
        clap_emb = F.normalize(self.clap.get_audio_embedding_from_filelist(x = audio_files, use_tensor=True))
        return clap_emb
    
    def emb_texts(self, texts:[str]):
        clap_emb = F.normalize(self.clap.get_text_embedding(x=texts, use_tensor=True))
        return clap_emb

# E14p
class EVA_CLIP_E14p(Expert):
    def __init__(self):
        super(EVA_CLIP_E14p, self).__init__()
        from eva_clip import create_model_and_transforms, get_tokenizer
        model_name = "EVA02-CLIP-bigE-14-plus" 
        pretrained = E14P_PATH # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"
        
        self.model, _, self.preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
        self.tokenizer = get_tokenizer(model_name)
        self.model.eval()
        
    @torch.no_grad()
    def emb_images(self, image_files:[str]):
        device = self.get_device()
        images = []
        for img_f in image_files:
            images.append(self.preprocess(Image.open(img_f).convert('RGB')))
            
        images = torch.stack(images).to(device)
        # print(images.shape)
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = F.normalize(image_features, dim=-1)
        return image_features
    
    @torch.no_grad()
    def emb_texts(self, texts:[str]):
        device = self.get_device()
        text = self.tokenizer(texts).to(device)
        text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    
    def get_device(self):
        return next(self.parameters()).device

# DFN5B
class DFN5B(Expert):
    def __init__(self):
        super(DFN5B, self).__init__()
        from open_clip import create_model_and_transforms, get_tokenizer
        model_name = 'ViT-H-14',  
        pretrained = DFN5B_PATH, # or "/path/to/DFN5B-CLIP-ViT-H-14/open_clip_pytorch_model.bin"
        
        self.model, self.preprocess = create_model_and_transforms(   
            model_name=model_name,  
            pretrained=pretrained,
            device = 'cpu'
        )
        self.tokenizer = get_tokenizer(model_name)
        self.model.eval()
        
    @torch.no_grad()
    def emb_images(self, image_files:[str]):
        device = self.get_device()
        images = []
        for img_f in image_files:
            images.append(self.preprocess(Image.open(img_f).convert('RGB')))
            
        images = torch.stack(images).to(device)
        # print(images.shape)
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = F.normalize(image_features, dim=-1)
        return image_features
    
    @torch.no_grad()
    def emb_texts(self, texts:[str]):
        device = self.get_device()
        text = self.tokenizer(texts, context_length=self.model.context_length).to(device)
        text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    
    def get_device(self):
        return next(self.parameters()).device

# EVA-18B
class EVA_18B(Expert):
    def __init__(self):
        super(EVA_18B, self).__init__()
        from transformers import AutoModel, CLIPImageProcessor, CLIPTokenizer
        model_name_or_path = "/root/autodl-tmp/huggingface_hub_down/BAAI/EVA-CLIP-18B" # or /path/to/local/EVA-CLIP-18B
        
        self.model = AutoModel.from_pretrained(
                    model_name_or_path, 
                    torch_dtype=torch.float16,
                    trust_remote_code=True).cpu().eval()
        self.preprocess = CLIPImageProcessor.from_pretrained("/root/autodl-tmp/huggingface_hub_down/openai/clip-vit-large-patch14")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path)
        self.model.eval()
        
    @torch.no_grad()
    def emb_images(self, image_files:[str]):
        device = self.get_device()
        images = []
        for img_f in image_files:
            images.append(self.processor(images=Image.open(img_f).convert('RGB'), 
                                         return_tensors="pt", padding=True).pixel_values)
        
        images = torch.stack(images).to(device)
        # print(images.shape)
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = F.normalize(image_features, dim=-1)
        return image_features
    
    @torch.no_grad()
    def emb_texts(self, texts:[str]):
        device = self.get_device()
        input_ids = self.tokenizer(texts, return_tensors = 'pt', 
                              padding= True,truncation=True,max_length=42).input_ids.to(device)
        text_features = self.model.encode_text(input_ids)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    
    def get_device(self):
        return next(self.model.parameters()).device

# SigLip384 SigLipso400m384
class SigLip384(Expert):
    def __init__(self, mode):
        super(EVA_18B, self).__init__()
        from transformers import AutoModel, AutoProcessor
        if mode == 'SIGLIP_384':
            model_name_or_path = SIGLIP_384 # or /path/to/local/siglip
        if mode == 'SIGLIP_400M':
            model_name_or_path = SIGLIP_400M
            
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        self.model.eval()
        
    @torch.no_grad()
    def emb_images(self, image_files:[str]):
        device = self.get_device()
        images = []
        for img_f in image_files:
            images.append(Image.open(img_f).convert('RGB'))
        input_data = self.processor(images=images, return_tensors="pt", padding="max_length")
        # images = torch.stack(images).to(device)
        # print(images.shape)
        input_data['pixel_values'] = input_data['pixel_values'].to(device)
        input_data['input_ids'] = input_data['input_ids'].to(device)
        # image = self.inputs = processor(text=texts, images=image, padding=, return_tensors="pt")
        image_features = self.model(**input_data).image_embeds
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = F.normalize(image_features, dim=-1)
        return image_features
    
    @torch.no_grad()
    def emb_texts(self, texts:[str]):
        device = self.get_device()
        input_data = self.processor(text=texts, padding="max_length", return_tensors="pt")['input_ids'].to(device)
        input_data['pixel_values'] = input_data['pixel_values'].to(device)
        input_data['input_ids'] = input_data['input_ids'].to(device)
        text_features = self.model(**input_data).text_embeds
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    
    def get_device(self):
        return next(self.model.parameters()).device


# Uni3D lvis mnet scan
def pc_norm(self, pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class Uni3D(nn.modules):
    def __init__(self, mode= 'lvis'):
        super(Uni3D, self).__init__()
        self.clip = EVA_CLIP_E14p()
        
        self.args = torch.load('')
        self.point_encoder = getattr(uni3d_models, self.args.model)(args=self.args).cpu()
        if mode == 'lvis':
            ckpt_path = LVIS_PATH
        if mode == 'mnet':
            ckpt_path = MNET_PATH
        if mode == 'scan':
            ckpt_path = SCAN_PATH
            
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        sd = checkpoint['module']
        self.point_encoder.load_state_dict(sd)
        print('load over')
        
    @torch.no_grad()
    def emb_images(self, image_files:[str]):
        device = self.clip.get_device()
        images = []
        for img_f in image_files:
            images.append(self.clip.preprocess(Image.open(img_f).convert('RGB')))
            
        images = torch.stack(images).to(device)
        # print(images.shape)
        image_features = self.clip.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = F.normalize(image_features, dim=-1)
        return image_features
    
    @torch.no_grad()
    def emb_texts(self, texts:[str]):
        device = self.clip.get_device()
        text = self.clip.tokenizer(texts).to(device)
        text_features = self.clip.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    
    @torch.no_grad()
    def emb_points(self, point_files:[str]):
        device = self.get_device()
        pc_gather  = []
        rgb_gather = []
        # switch to evaluate mode
        for pf in point_files:
            obj = np.load(pf, allow_pickle=True).item()
    
            pc = obj['xyz'].astype(np.float32)
            rgb = obj['rgb'].astype(np.float32)
            pc = pc_norm(pc)
            pc, rgb = torch.from_numpy(pc).float(), torch.from_numpy(rgb).float()
            pc_gather.append(pc)
            rgb_gather.append(rgb)
        
        pc_gather  = torch.cat(pc_gather, dim=0)
        rgb_gather = torch.cat(rgb_gather, dim=0)

        with torch.no_grad():
            pc = pc_gather.to(device=device, non_blocking=True)
            rgb = rgb_gather.to(device=device, non_blocking=True)
            feature = torch.cat((pc, rgb),dim=-1)

            # encode pc
            pc_features = uni3d_utils.get_model(self.point_encoder).encode_pc(feature)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
            # pc_embs.append(pc_features)
            # pc_embs = torch.cat(pc_embs)
            # pc_embs = F.normalize(pc_embs, dim=-1)
            return pc_features
    
    def get_device(self):
        return next(self.point_encoder.parameters()).device