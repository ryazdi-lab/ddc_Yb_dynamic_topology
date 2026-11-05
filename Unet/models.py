"""Fixed three-slot, paired-parameter models for controlled input ablations."""
import hashlib
import torch
from torch import nn
from unet_model import UNet
from attention_unet_model import AttentionUNet
from dataset import VARIANTS


class AblationModel(nn.Module):
    def __init__(self,model_type,variant,target='dq',base_channels=32,attn_heads=4):
        super().__init__()
        mask=torch.zeros(3);mask[VARIANTS[variant]]=1
        self.register_buffer('input_mask',mask.reshape(1,3,1,1))
        mode='matrix' if target=='dq' else 'vector'
        if model_type=='unet':self.network=UNet(base_channels=base_channels,output_mode=mode,in_channels=3)
        elif model_type=='attention_unet':self.network=AttentionUNet(base_channels=base_channels,attn_heads=attn_heads,output_mode=mode,in_channels=3)
        else:raise ValueError('Unknown model type')
    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=3:raise ValueError('Expected normalized X with three fixed slots')
        return self.network(x*self.input_mask)


def parameter_hash(model):
    h=hashlib.sha256()
    for p in model.parameters():h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
