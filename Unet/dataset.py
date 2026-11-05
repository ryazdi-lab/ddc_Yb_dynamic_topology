"""Strict loader for dd c/Yb train-normalized v2 NPZ files. No refitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

VARIANTS={'ddc':[0],'ddc_yb':[0,1,2],'ddc_imag':[0,2],'ddc_real':[0,1]}
STAT_KEYS=['yb_asinh_scale','input_center','input_scale','dq_center','dq_scale','vmag_center','vmag_scale']


def stats_signature(stats):
    return hashlib.sha256(json.dumps({k:stats[k] for k in STAT_KEYS},sort_keys=True).encode()).hexdigest()


class DependencyMatrixDataset(Dataset):
    """Return normalized full-slot X, normalized target, and raw target.

    Excluded input slots are masked inside AblationModel, after normalization.
    Raw targets are used only for metrics, never model inputs.
    """
    def __init__(self,npz_path,target_key='dq',limit=None):
        self.path=Path(npz_path);self.target_key=target_key.lower()
        if self.target_key not in ['dq','vmag']:raise ValueError('target_key must be dq or vmag')
        with np.load(self.path,allow_pickle=False) as z:
            if str(z['schema_version'])!='ddc_yb_train_normalized_v2':raise ValueError('Use dl_normalized_datasets v2 files')
            self.stats={k:z[k].tolist() for k in STAT_KEYS}
            self.signature=stats_signature(self.stats)
            x=z['X'];y=z['y' if self.target_key=='dq' else 'Vmag']
            raw=z['DQ_raw' if self.target_key=='dq' else 'Vmag_raw']
            ids=z['source_indices']
        if limit is not None:x=x[:limit];y=y[:limit];raw=raw[:limit];ids=ids[:limit]
        if x.ndim!=4 or x.shape[1]!=3 or x.shape[2]!=x.shape[3]:raise ValueError(f'Expected X (N,3,B,B), got {x.shape}')
        self.buses=x.shape[-1]
        expected=(len(x),1,self.buses,self.buses) if self.target_key=='dq' else (len(x),self.buses)
        if y.shape!=expected or raw.shape!=expected:raise ValueError('Target geometry does not match task')
        if len(x)<2 or len(ids)!=len(x) or len(np.unique(ids))!=len(ids):raise ValueError('Invalid sample indices/count')
        if not all(np.isfinite(a).all() for a in [x,y,raw]):raise ValueError('Nonfinite dataset values')
        for key in ['input_scale','dq_scale','vmag_scale','yb_asinh_scale']:
            if (np.asarray(self.stats[key])<=0).any():raise ValueError(f'Invalid {key}')
        self.X=torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32))
        self.y=torch.from_numpy(np.ascontiguousarray(y,dtype=np.float32))
        self.raw=torch.from_numpy(np.ascontiguousarray(raw,dtype=np.float64))
        self.ids=np.array(ids,copy=True)
        # Catch targets normalized with inconsistent parameters.
        if not torch.allclose(self.denormalize_y(self.y).double(),self.raw,rtol=2e-5,atol=1e-5):
            raise ValueError('Stored target and normalization do not agree')

    def __len__(self):return len(self.X)
    def __getitem__(self,i):return self.X[i],self.y[i],self.raw[i]
    def denormalize_y(self,value):
        if self.target_key=='dq':center,scale=self.stats['dq_center'],self.stats['dq_scale']
        else:center,scale=self.stats['vmag_center'],self.stats['vmag_scale']
        return value*torch.as_tensor(scale,device=value.device,dtype=value.dtype)+torch.as_tensor(center,device=value.device,dtype=value.dtype)
    def stats_dict(self):return dict(self.stats)
    def __repr__(self):return f'Dataset({self.target_key}, X={tuple(self.X.shape)}, y={tuple(self.y.shape)})'


def check_splits(*datasets):
    first=datasets[0];seen=set()
    for ds in datasets:
        if ds.buses!=first.buses or ds.signature!=first.signature:raise ValueError('Split normalization/bus ordering differs')
        ids=set(ds.ids.tolist())
        if seen&ids:raise ValueError('Overlapping source samples across splits')
        seen|=ids


def get_dataloaders(train_path,val_path,test_path,batch_size=64,num_workers=0,target_key='dq'):
    """Convenience function; the runner defers test loading until selection ends."""
    datasets=[DependencyMatrixDataset(p,target_key) for p in [train_path,val_path,test_path]]
    check_splits(*datasets)
    return tuple(DataLoader(ds,batch_size=batch_size,shuffle=i==0,num_workers=num_workers) for i,ds in enumerate(datasets))
