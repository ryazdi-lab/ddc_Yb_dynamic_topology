"""Global pooled metrics in clearly labelled target coordinates."""
import numpy as np


def compute_metrics(pred,target):
    p,t=np.asarray(pred,dtype=float),np.asarray(target,dtype=float)
    if p.shape!=t.shape or not np.isfinite(p).all() or not np.isfinite(t).all():raise ValueError('Invalid predictions/targets')
    e=p-t;sst=np.square(t-t.mean()).sum()
    return {'r2':float(1-np.square(e).sum()/sst) if sst>1e-20 else None,
            'rmse':float(np.sqrt(np.square(e).mean())),'mae':float(np.abs(e).mean())}
