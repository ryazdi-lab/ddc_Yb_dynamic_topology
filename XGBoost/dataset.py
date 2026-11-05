"""Strict normalized v2 loader; independent of PyTorch."""
import hashlib
import json
from pathlib import Path
import numpy as np

VARIANTS={'ddc':[0],'ddc_yb':[0,1,2],'ddc_imag':[0,2],'ddc_real':[0,1]}
STAT_KEYS=['yb_asinh_scale','input_center','input_scale','dq_center','dq_scale','vmag_center','vmag_scale']


def signature(stats):
    return hashlib.sha256(json.dumps({k:stats[k] for k in STAT_KEYS},sort_keys=True).encode()).hexdigest()


def load(path,target='dq',limit=None):
    with np.load(path,allow_pickle=False) as z:
        if str(z['schema_version'])!='ddc_yb_train_normalized_v2':raise ValueError('Use dl_normalized_datasets v2 files')
        x=z['X'];y=z['y' if target=='dq' else 'Vmag'];raw=z['DQ_raw' if target=='dq' else 'Vmag_raw']
        ids=z['source_indices'];stats={k:z[k].tolist() for k in STAT_KEYS}
    if limit is not None:x=x[:limit];y=y[:limit];raw=raw[:limit];ids=ids[:limit]
    if x.ndim!=4 or x.shape[1]!=3 or x.shape[2]!=x.shape[3]:raise ValueError('Expected X (N,3,B,B)')
    n=x.shape[-1];expected=(len(x),1,n,n) if target=='dq' else (len(x),n)
    if y.shape!=expected or raw.shape!=expected:raise ValueError('Target layout mismatch')
    if len(x)<2 or not all(np.isfinite(a).all() for a in [x,y,raw]):raise ValueError('Invalid values/sample count')
    if len(np.unique(ids))!=len(ids):raise ValueError('Duplicate source samples')
    return {'X':np.ascontiguousarray(x,dtype=np.float32),'y':y.reshape(len(x),-1).astype(np.float32),
            'raw':raw.reshape(len(x),-1),'ids':ids,'stats':stats,'signature':signature(stats),
            'buses':n,'target':target,'target_shape':list(expected[1:])}


def features(x,variant):
    """Only declared channels are supplied to the booster; compact flattened layout."""
    return np.ascontiguousarray(x[:,VARIANTS[variant]].reshape(len(x),-1),dtype=np.float32)


def check_splits(*splits):
    base=splits[0];seen=set()
    for s in splits:
        if s['buses']!=base['buses'] or s['signature']!=base['signature']:raise ValueError('Split normalization/bus count mismatch')
        ids=set(s['ids'].tolist())
        if seen&ids:raise ValueError('Overlapping source samples')
        seen|=ids


def decode(pred,stats,target,indices):
    p=np.asarray(pred,dtype=float)
    if target=='dq':return p*stats['dq_scale']+stats['dq_center']
    return p*np.asarray(stats['vmag_scale'])[indices]+np.asarray(stats['vmag_center'])[indices]
