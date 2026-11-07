"""Train-fitted normalization for Ddc/Yb -> DQ/Vmag; NumPy only.

Input: topology_ablation_datasets/case*/ from convert_mat_ablation.py.
Output: float32 training arrays plus lossless float64 raw data and metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

CHANNELS={'ddc':[0],'ddc_yb':[0,1,2],'ddc_imag':[0,2],'ddc_real':[0,1]}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read_raw(path):
    with np.load(path,allow_pickle=False) as z:
        if str(z['schema_version'])!='ddc_yb_ablation_v1':raise ValueError('Unexpected input schema')
        x=z['X']
        x[:,0]*=z['Ddc_scale'][:,None,None]
        y=z['y']*z['DQ_scale'][:,None,None,None]
        return x,y,z['Vmag'],z['source_indices']


def safe_std(value,floor):
    a=np.asarray(value,dtype=float)
    return np.where(a<floor,1.0,a)


def fitted_stats(x,y,v,source_ids):
    """Fit exclusively on the training arrays; no validation/test arguments."""
    if not all(np.isfinite(a).all() for a in [x,y,v]):raise ValueError('Nonfinite training values')
    s=np.ones(3,dtype=float)
    for c in [1,2]:
        values=np.abs(x[:,c]);nonzero=values[values>0]
        s[c]=float(np.median(nonzero)) if nonzero.size else 1.0
        s[c]=max(s[c],1e-12)
    center=[];scale=[]
    for c in range(3):
        a=x[:,c] if c==0 else np.arcsinh(x[:,c]/s[c])
        center.append(float(a.mean(dtype=float)))
        scale.append(float(safe_std(a.std(dtype=float),1e-12)))
    return {'schema_version':'ddc_yb_train_normalized_v2','fit_split':'train',
            'fit_sample_count':len(x),'fit_source_indices_sha256':hashlib.sha256(source_ids.astype('<i8').tobytes()).hexdigest(),
            'channels':['Ddc','Re_Yb','Im_Yb'],'yb_asinh_scale':s.tolist(),
            'input_center':center,'input_scale':scale,
            'dq_center':float(y.mean(dtype=float)),'dq_scale':float(safe_std(y.std(dtype=float),1e-12)),
            'vmag_center':v.mean(axis=0,dtype=float).tolist(),
            'vmag_scale':safe_std(v.std(axis=0,dtype=float),1e-6).tolist(),
            'constant_scale_rule':'std < 1e-12 (input/DQ) or < 1e-6 (Vmag per bus) uses scale=1',
            'clipping':False,'float_dtype':'float32',
            'dq_target':'RAW DQ standardized with one scalar mean/std fitted on train; no per-sample target scale'}


def encode_x(x,stats):
    x=np.asarray(x,dtype=float)
    if x.ndim!=4 or x.shape[1]!=3:raise ValueError('Expected raw X (N,3,B,B)')
    out=np.empty(x.shape,dtype=np.float32)
    for c in range(3):
        a=x[:,c] if c==0 else np.arcsinh(x[:,c]/stats['yb_asinh_scale'][c])
        out[:,c]=(a-stats['input_center'][c])/stats['input_scale'][c]
    return out


def decode_x(x,stats):
    out=np.asarray(x,dtype=float).copy()
    for c in range(3):
        a=out[:,c]*stats['input_scale'][c]+stats['input_center'][c]
        out[:,c]=a if c==0 else np.sinh(a)*stats['yb_asinh_scale'][c]
    return out


def inverse_dq(prediction,stats):
    """Accept NumPy predictions after tensor.detach().cpu().numpy()."""
    return np.asarray(prediction,dtype=float)*stats['dq_scale']+stats['dq_center']


def inverse_vmag(prediction,stats):
    return np.asarray(prediction,dtype=float)*np.asarray(stats['vmag_scale'])+np.asarray(stats['vmag_center'])


def transform_inputs(Ddc,Yb,stats,variant='ddc_yb'):
    """Inference from raw matrices. No DQ, Vmag, or unknown target scales needed.

    Ddc is required for all four variants. Yb is required for Yb variants.
    For ddc_real or ddc_imag, Yb can be complex full Yb or a real array of
    just the required component. Components not used by that variant are zero.
    """
    dc=np.asarray(Ddc,dtype=float)
    if dc.ndim==2:dc=dc[None]
    if dc.ndim!=3 or dc.shape[1]!=dc.shape[2]:raise ValueError('Expected Ddc (N,B,B) or (B,B)')
    x=np.zeros((len(dc),3,*dc.shape[1:]),dtype=float);x[:,0]=dc
    if variant!='ddc':
        if Yb is None:raise ValueError('This variant requires Yb')
        yb=np.asarray(Yb)
        if yb.ndim==2:yb=yb[None]
        if yb.shape!=dc.shape:raise ValueError('Ddc and Yb shapes differ')
        if variant=='ddc_yb':
            if not np.iscomplexobj(yb):raise ValueError('Full Yb requires complex data with real and imaginary parts')
            x[:,1]=yb.real;x[:,2]=yb.imag
        elif variant=='ddc_real':x[:,1]=yb.real
        elif variant=='ddc_imag':x[:,2]=yb.imag if np.iscomplexobj(yb) else yb
    if not np.isfinite(x).all():raise ValueError('Nonfinite input')
    return encode_x(x,stats)[:,CHANNELS[variant]]


def load_ablation(path,variant):
    with np.load(path,allow_pickle=False) as z:
        if str(z['schema_version'])!='ddc_yb_train_normalized_v2':raise ValueError('Unexpected dataset schema')
        return z['X'][:,CHANNELS[variant]],{'DQ':z['y'],'Vmag':z['Vmag']}


def convert_case(source,out):
    source,out=Path(source),Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a fresh empty output directory')
    train_path=source/'dataset_train.npz'
    x,y,v,ids=read_raw(train_path)
    stats=fitted_stats(x,y,v,ids)
    stats['training_file_sha256']=sha256(train_path)
    stats['training_file']=str(train_path.resolve())
    out.mkdir(parents=True,exist_ok=True)
    (out/'normalization.json').write_text(json.dumps(stats,indent=2))
    del x,y,v,ids
    manifest={'normalization':'normalization.json','ablation_channels':CHANNELS,'splits':{}}
    all_ids=[]
    for split in ['train','val','test']:
        x,y,v,ids=read_raw(source/f'dataset_{split}.npz')
        xn=encode_x(x,stats)
        yn=((y-stats['dq_center'])/stats['dq_scale']).astype(np.float32)
        vn=((v-np.array(stats['vmag_center']))/np.array(stats['vmag_scale'])).astype(np.float32)
        # Topology masks are defined from raw Yb, never centered/scaled values.
        adjacency=(x[:,1]!=0)|(x[:,2]!=0)
        adjacency[:,np.arange(x.shape[-1]),np.arange(x.shape[-1])]=False
        adjacency|=adjacency.transpose(0,2,1)
        arrays={'X':xn,'y':yn,'Vmag':vn,'X_raw':x,'DQ_raw':y,'Vmag_raw':v,
                'adjacency':adjacency,'source_indices':ids,
                'channel_names':np.array(stats['channels']),
                'schema_version':np.array(stats['schema_version'])}
        for key in ['yb_asinh_scale','input_center','input_scale','dq_center','dq_scale','vmag_center','vmag_scale']:
            arrays[key]=np.array(stats[key],dtype=float)
        assert all(np.isfinite(a).all() for a in [xn,yn,vn,x,y,v])
        dqerr=float(np.abs(inverse_dq(yn,stats)-y).max())
        verr=float(np.abs(inverse_vmag(vn,stats)-v).max())
        back=decode_x(xn,stats)
        # Relative + absolute tolerance accounts for float32 normalization.
        assert np.allclose(back,x,rtol=2e-5,atol=2e-5)
        assert dqerr<1e-4 and verr<1e-6
        max_xerr=float(np.max(np.abs(back-x)/np.maximum(1,np.abs(x))))
        del back
        if split=='train':
            assert np.allclose(xn.mean((0,2,3),dtype=float),0,atol=1e-6)
            assert np.allclose(xn.std((0,2,3),dtype=float),1,atol=1e-6)
            assert abs(float(yn.mean(dtype=float)))<1e-6
            assert abs(float(yn.std(dtype=float))-1)<1e-6
        # Raw inference preprocessing must match every selected saved channel.
        for variant in CHANNELS:
            small=transform_inputs(x[:8,0],x[:8,1]+1j*x[:8,2],stats,variant)
            assert np.array_equal(small,xn[:8,CHANNELS[variant]])
        path=out/f'dataset_{split}.npz'
        np.savez_compressed(path,**arrays)
        with np.load(path,allow_pickle=False) as z:
            for key,a in arrays.items():assert np.array_equal(z[key],a),key
            assert z['X'].dtype==z['y'].dtype==z['Vmag'].dtype==np.float32
        all_ids.extend(ids.tolist())
        manifest['splits'][split]={'samples':len(ids),'source_indices_0based':ids.tolist(),
            'file':path.name,'sha256':sha256(path),'X_shape':list(xn.shape),
            'y_shape':list(yn.shape),'Vmag_shape':list(vn.shape),
            'max_dq_inverse_error':dqerr,'max_vmag_inverse_error':verr,
            'max_input_inverse_error_scaled_by_max1abs':max_xerr}
        print(source.name,split,len(ids),'DQ inverse error',dqerr,'Vmag inverse error',verr,flush=True)
        del arrays,x,y,v,xn,yn,vn,adjacency
    assert len(set(all_ids))==len(all_ids)
    # Independent refit on train confirms recorded statistics depend only on train.
    x,y,v,ids=read_raw(train_path)
    refit=fitted_stats(x,y,v,ids)
    assert all(stats[k]==val for k,val in refit.items())
    del x,y,v,ids
    manifest['verification']='Passed train-only fit/refit, all split read-backs, raw preservation, inverse transforms, four inference channel selections and disjoint source assignments.'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Verified',out.resolve(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--out-dir',required=True)
    a=p.parse_args();convert_case(a.source,a.out_dir)
