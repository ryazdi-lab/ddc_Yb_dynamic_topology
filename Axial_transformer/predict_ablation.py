"""Inference from normalized NPZ or raw Ddc/Yb, using saved preprocessing."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dataset import VARIANTS,STAT_KEYS,stats_signature
from models import AblationModel
from normalization import transform_inputs,inverse_dq,inverse_vmag
from train_transformer_imlive import device_for


def load_checkpoint(checkpoint,device):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    meta=state['metadata']
    if meta['schema_version']!='ddc_yb_train_normalized_v2':raise ValueError('Unsupported checkpoint schema')
    model=AblationModel(**meta['model_config']).to(device)
    model.load_state_dict(state['state']);model.eval()
    return model,meta


@torch.inference_mode()
def run_batches(model,meta,x,device,batch_size):
    if batch_size<1:raise ValueError('batch_size must be positive')
    b=meta['buses']
    if x.ndim!=4 or x.shape[1:]!=(3,b,b) or len(x)==0:raise ValueError('Input layout/bus count differs from checkpoint')
    if not np.isfinite(x).all():raise ValueError('Nonfinite inputs')
    pred=np.concatenate([model(torch.as_tensor(x[i:i+batch_size],device=device,dtype=torch.float32)).float().cpu().numpy()
                         for i in range(0,len(x),batch_size)])
    inverse=inverse_dq if meta['model_config']['target']=='dq' else inverse_vmag
    return inverse(pred,meta['normalization'])


def predict_raw(checkpoint,Ddc,Yb=None,device='auto',batch_size=64):
    """Inputs (N,B,B) or (B,B). Returns raw DQ (N,1,B,B) or Vmag (N,B).

    Yb may be omitted for ddc. Partial variants accept their single real-valued
    component or a full complex matrix. No target values/scales are needed.
    """
    device=device_for(device);model,meta=load_checkpoint(checkpoint,device)
    variant=meta['model_config']['variant']
    selected=transform_inputs(Ddc,Yb,meta['normalization'],variant)
    x=np.zeros((len(selected),3,*selected.shape[-2:]),dtype=np.float32)
    x[:,VARIANTS[variant]]=selected
    return run_batches(model,meta,x,device,batch_size)


def predict_npz(checkpoint,path,device='auto',batch_size=64):
    device=device_for(device);model,meta=load_checkpoint(checkpoint,device)
    with np.load(path,allow_pickle=False) as z:
        if str(z['schema_version'])!=meta['schema_version']:raise ValueError('Dataset schema mismatch')
        stats={k:z[k].tolist() for k in STAT_KEYS}
        if stats_signature(stats)!=meta['normalization_signature']:raise ValueError('Normalization differs from checkpoint')
        x=z['X']
    return run_batches(model,meta,x,device,batch_size)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--input',required=True)
    p.add_argument('--output',default='predictions.npz');p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--cpu-threads',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(a.cpu_threads)
    out=Path(a.output)
    if out.exists():raise FileExistsError(f'{out} exists; choose another output file')
    if out.resolve()==Path(a.input).resolve():raise ValueError('Output must differ from input')
    pred=predict_npz(a.checkpoint,a.input,a.device,a.batch_size)
    np.savez_compressed(out,prediction=pred)
    print('Saved physical-unit predictions',pred.shape,out.resolve())
