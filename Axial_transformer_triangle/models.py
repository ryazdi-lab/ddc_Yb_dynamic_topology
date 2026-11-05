"""Fixed three-slot wrappers (ddc_imag_pinv receives prepared pinv features) around the transformer architectures.

head='plain'          network output is the standardized target (original behaviour).
head='residual_gate'  DQ only. Output = where(Ddc != 0, a*Ddc + b + network, zero_value):
                      entries where Ddc is exactly zero are exactly zero in DQ, so they are
                      set to the standardized-zero constant; elsewhere the network learns a
                      correction on top of a learned linear map of Ddc. The network's last
                      layer starts at zero, so training starts from the linear-gate baseline.
All constants are fitted from the training split only (head_constants).
"""
import hashlib
import torch
from torch import nn
from dataset import VARIANTS
from transformer_model import DQTransformer,AxialTransformerDQNet,DQTransformerVmagNet,AxialTransformerVmagNet
from triangle_model import TriangleDQNet,TriangleVmagNet

# Float32 normalization collapses raw Ddc below ~3e-7 onto the same value as an exact zero,
# so the "is zero" test uses a few float32 ulps of tolerance in standardized units.
GATE_TOL=1e-6
ARCHITECTURES={'axial':(AxialTransformerDQNet,AxialTransformerVmagNet),
               'dq':(DQTransformer,DQTransformerVmagNet),
               'tri':(TriangleDQNet,TriangleVmagNet)}


def ddc_zero_value(stats):
    """Standardized value of a raw Ddc entry equal to 0."""
    return -stats['input_center'][0]/stats['input_scale'][0]


def nonzero_mask(ddc_normalized,stats,tol=GATE_TOL):
    """True where raw Ddc is nonzero. Input: normalized Ddc slot (any shape)."""
    return (ddc_normalized-ddc_zero_value(stats)).abs()>tol


def head_constants(train,tol=GATE_TOL):
    """Gate/skip constants for head='residual_gate', from the training split only."""
    if train.target_key!='dq':raise ValueError('residual_gate head is defined for the dq target only')
    s=train.stats;ddc_zero=ddc_zero_value(s);dq_zero=-s['dq_center']/s['dq_scale']
    x=train.X[:,:1].double();y=train.y.double()
    keep=(x-ddc_zero).abs()>tol
    off=(y[~keep]-dq_zero).abs()
    if off.numel() and float((off>1e-2).double().mean())>1e-4:
        raise ValueError('Gate assumption failed: DQ is not zero where Ddc is zero in this dataset; use --head plain')
    xs,ys=x[keep],y[keep];xm,ym=xs.mean(),ys.mean()
    a=float(((xs-xm)*(ys-ym)).mean()/((xs-xm)**2).mean());b=float(ym-a*xm)
    return {'ddc_zero':float(ddc_zero),'dq_zero':float(dq_zero),'skip_a':a,'skip_b':b}


class AblationModel(nn.Module):
    def __init__(self,model_type,variant,target='dq',n_buses=30,d_model=96,depth=4,num_heads=6,dropout=.1,mlp_ratio=4.,
                 head='plain',ddc_zero=None,dq_zero=None,skip_a=1.,skip_b=0.,gate_tol=GATE_TOL):
        super().__init__()
        mask=torch.zeros(3);mask[VARIANTS[variant]]=1
        self.register_buffer('input_mask',mask.reshape(1,3,1,1))
        if target not in ['dq','vmag']:raise ValueError('Target must be dq or vmag')
        if model_type not in ARCHITECTURES:raise ValueError('Supported architectures: axial, dq (global) or tri')
        if head not in ['plain','residual_gate']:raise ValueError('head must be plain or residual_gate')
        if head=='residual_gate' and (target!='dq' or ddc_zero is None or dq_zero is None):
            raise ValueError('residual_gate needs target=dq plus ddc_zero and dq_zero constants')
        cls=ARCHITECTURES[model_type][0 if target=='dq' else 1]
        self.network=cls(n_buses=n_buses,d_model=d_model,depth=depth,num_heads=num_heads,
                         dropout=dropout,mlp_ratio=mlp_ratio,in_channels=3)
        self.head=head;self.ddc_zero=ddc_zero;self.dq_zero=dq_zero;self.gate_tol=gate_tol
        if head=='residual_gate':
            self.skip_a=nn.Parameter(torch.tensor(float(skip_a)))
            self.skip_b=nn.Parameter(torch.tensor(float(skip_b)))
            last=self.network.out.proj[-1]   # start exactly at the gated linear baseline
            nn.init.zeros_(last.weight);nn.init.zeros_(last.bias)
    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=3:raise ValueError('Expected normalized X (N,3,B,B)')
        out=self.network(x*self.input_mask)
        if self.head=='plain':return out
        ddc=x[:,:1]   # slot 0 (Ddc) is present in every variant
        keep=(ddc-self.ddc_zero).abs()>self.gate_tol
        return torch.where(keep,out+self.skip_a*ddc+self.skip_b,self.dq_zero)


def parameter_hash(model):
    h=hashlib.sha256()
    for p in model.parameters():h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
