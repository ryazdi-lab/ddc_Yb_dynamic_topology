"""Derived input: float64 pinv(raw Im(Yb)), with train-only feature scaling.

Prepared matrices replace slot 2 only for ddc_imag_pinv. The model still has
three slots, and neither DQ nor Vmag is used to construct this feature.
"""
import copy
import hashlib
import numpy as np
import torch


def pseudoinverse(imag,rcond=1e-8,chunk_size=64):
    a=np.asarray(imag,dtype=np.float64)
    if a.ndim!=3 or a.shape[1]!=a.shape[2] or len(a)==0:raise ValueError('Expected raw Im(Yb) shape (N,B,B)')
    if not 0<rcond<1 or chunk_size<1:raise ValueError('Invalid pseudoinverse settings')
    if not np.isfinite(a).all():raise ValueError('Nonfinite raw admittance')
    out=np.empty_like(a)
    for i in range(0,len(a),chunk_size):
        out[i:i+chunk_size]=np.linalg.pinv(a[i:i+chunk_size],rcond=rcond,hermitian=False)
    if not np.isfinite(out).all():raise ValueError('Nonfinite pseudoinverse feature')
    return out


def raw_imag(ds):
    with np.load(ds.path,allow_pickle=False) as z:
        if 'X_raw' not in z:raise ValueError('ddc_imag_pinv requires X_raw; inverse of normalized Im(Yb) is not valid')
        x=z['X_raw']
        if x.ndim!=4 or x.shape[1]!=3:raise ValueError('Expected X_raw (N,3,B,B)')
        out=np.asarray(x[:len(ds),2],dtype=np.float64).copy()
        if not np.array_equal(z['source_indices'][:len(ds)],ds.ids):raise ValueError('Raw feature/sample alignment differs')
    return out


def fit_scaler(p,rcond,source_ids,base_signature):
    """Training feature array only. No targets or held-out inputs accepted."""
    nz=np.abs(p[p!=0]);s=max(float(np.median(nz)) if nz.size else 1.,1e-12)
    t=np.arcsinh(p/s);center=float(t.mean(dtype=float));scale=float(t.std(dtype=float))
    if scale<1e-12:scale=1.
    return {'kind':'pinv_raw_imag_v1','slot':2,'rcond':float(rcond),'hermitian':False,
            'asinh_scale':s,'center':center,'scale':scale,'fit_split':'train',
            'fit_sample_count':len(p),'base_normalization_signature':base_signature,
            'fit_source_indices_sha256':hashlib.sha256(source_ids.astype('<i8').tobytes()).hexdigest()}


def encode(p,stats):
    out=((np.arcsinh(p/stats['asinh_scale'])-stats['center'])/stats['scale']).astype(np.float32)
    if not np.isfinite(out).all():raise ValueError('Nonfinite encoded pseudoinverse')
    return out


def encode_raw_imag(imag,stats,chunk_size=64):
    if stats.get('kind')!='pinv_raw_imag_v1':raise ValueError('Unsupported pseudoinverse transform')
    return encode(pseudoinverse(imag,stats['rcond'],chunk_size),stats)


def prepared_copy(ds,encoded,stats):
    if stats['base_normalization_signature']!=ds.signature:raise ValueError('Input normalization does not match pinv training fit')
    if encoded.shape!=(len(ds),ds.buses,ds.buses):raise ValueError('Derived feature geometry differs')
    result=copy.copy(ds)
    result.X=ds.X.clone();result.X[:,2]=torch.from_numpy(encoded)
    result.input_transform=copy.deepcopy(stats)
    return result


def prepare_training(ds,rcond=1e-8,chunk_size=64):
    p=pseudoinverse(raw_imag(ds),rcond,chunk_size)
    stats=fit_scaler(p,rcond,ds.ids,ds.signature)
    return prepared_copy(ds,encode(p,stats),stats),stats


def prepare_split(ds,stats,chunk_size=64):
    return prepared_copy(ds,encode_raw_imag(raw_imag(ds),stats,chunk_size),stats)
