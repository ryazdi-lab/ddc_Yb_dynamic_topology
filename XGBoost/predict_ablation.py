"""Predict original-unit DQ/Vmag with saved XGBoost ablation models."""
import argparse
import json
from pathlib import Path
import numpy as np
from dataset import VARIANTS,STAT_KEYS,signature,decode
from normalization import transform_inputs
from gpu_xgboost_vmag import predict_normalized,validate_device


def prediction_shape(raw,meta):
    return raw.reshape(len(raw),*meta['target_shape']) if meta['complete_target'] else raw


def predict_raw(model_dir,Ddc,Yb=None,device='cpu',threads=4):
    """Raw arrays (N,B,B) or (B,B); no target values/scales are needed."""
    folder=Path(model_dir);meta=json.loads((folder/'metadata.json').read_text())
    validate_device(device,threads)
    variant=meta['variant'];selected=transform_inputs(Ddc,Yb,meta['normalization'],variant)
    x=np.zeros((len(selected),3,*selected.shape[-2:]),dtype=np.float32);x[:,VARIANTS[variant]]=selected
    pp,meta=predict_normalized(folder,[x],device,threads)
    return prediction_shape(decode(pp[0],meta['normalization'],meta['target'],meta['output_indices']),meta)


def predict_npz(model_dir,path,device='cpu',threads=4):
    folder=Path(model_dir);meta=json.loads((folder/'metadata.json').read_text());validate_device(device,threads)
    with np.load(path,allow_pickle=False) as z:
        if str(z['schema_version'])!=meta['schema_version']:raise ValueError('Dataset schema differs')
        stats={k:z[k].tolist() for k in STAT_KEYS}
        if signature(stats)!=meta['normalization_signature']:raise ValueError('Normalization differs')
        x=z['X']
    pp,meta=predict_normalized(folder,[x],device,threads)
    return prediction_shape(decode(pp[0],meta['normalization'],meta['target'],meta['output_indices']),meta)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--input',required=True)
    p.add_argument('--output',default='xgboost_predictions.npz');p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Choose a new output path')
    pred,meta=predict_npz(a.model_dir,a.input,a.device,a.threads)
    np.savez_compressed(out,prediction=pred,output_indices=np.array(meta['output_indices']))
    print('Saved original-unit predictions',pred.shape,out.resolve())
