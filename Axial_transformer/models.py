"""Fixed three-slot wrappers around the attached transformer architectures."""
import hashlib
import torch
from torch import nn
from dataset import VARIANTS
from transformer_model import DQTransformer,AxialTransformerDQNet,DQTransformerVmagNet,AxialTransformerVmagNet


class AblationModel(nn.Module):
    def __init__(self,model_type,variant,target='dq',n_buses=30,d_model=96,depth=4,num_heads=6,dropout=.1,mlp_ratio=4.):
        super().__init__()
        mask=torch.zeros(3);mask[VARIANTS[variant]]=1
        self.register_buffer('input_mask',mask.reshape(1,3,1,1))
        if target not in ['dq','vmag']:raise ValueError('Target must be dq or vmag')
        if model_type=='axial':cls=AxialTransformerDQNet if target=='dq' else AxialTransformerVmagNet
        elif model_type=='dq':cls=DQTransformer if target=='dq' else DQTransformerVmagNet
        else:raise ValueError('Supported architectures: axial or dq (global)')
        self.network=cls(n_buses=n_buses,d_model=d_model,depth=depth,num_heads=num_heads,
                         dropout=dropout,mlp_ratio=mlp_ratio,in_channels=3)
    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=3:raise ValueError('Expected normalized X (N,3,B,B)')
        return self.network(x*self.input_mask)


def parameter_hash(model):
    h=hashlib.sha256()
    for p in model.parameters():h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
