"""Pseudo-inverse math, train-only feature preparation and checkpoint inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from dataset import DependencyMatrixDataset,STAT_KEYS
from pinv_features import pseudoinverse,prepare_training,prepare_split,encode_raw_imag
from predict_ablation import predict_raw,predict_npz


def run(data_dir,smoke_dir):
    torch.set_num_threads(4);report={}
    lap=3*np.eye(3)-np.ones((3,3))
    assert np.allclose(pseudoinverse(lap[None])[0],lap/9,atol=1e-14)
    assert not pseudoinverse(np.zeros((1,3,3))).any()
    u=np.array([1.,2.,3.]);v=np.array([2.,-1.,4.]);a=np.outer(u,v)
    want=np.outer(v,u)/((u@u)*(v@v))
    assert np.allclose(pseudoinverse(a[None])[0],want,atol=1e-14)
    diag=np.diag([4.,1e-10,0.])[None]
    assert np.allclose(pseudoinverse(diag,1e-8)[0],np.diag([.25,0,0]),atol=1e-14)
    assert np.isclose(pseudoinverse(diag,1e-12)[0,1,1],1e10)
    report['analytic_singular_zero_nonsymmetric_and_cutoff_checks']=True
    root=Path(data_dir)
    train=DependencyMatrixDataset(root/'dataset_train.npz','dq',16)
    val=DependencyMatrixDataset(root/'dataset_val.npz','dq',8)
    original=train.X.clone();prepared,stats=prepare_training(train)
    assert torch.equal(train.X,original) and torch.equal(prepared.X[:,0],train.X[:,0])
    assert prepared.y is train.y and prepared.raw is train.raw
    x=prepared.X[:,2].numpy()
    assert abs(x.mean(dtype=float))<1e-6 and abs(x.std(dtype=float)-1)<1e-6
    assert stats['fit_source_indices_sha256']==hashlib.sha256(train.ids.astype('<i8').tobytes()).hexdigest()
    corrupt=copy.copy(train);corrupt.y=torch.randn_like(train.y);corrupt.raw=torch.ones_like(train.raw)*1e9
    other,other_stats=prepare_training(corrupt)
    assert stats==other_stats and torch.equal(prepared.X,other.X)
    frozen=copy.deepcopy(stats);held_out=prepare_split(val,stats)
    assert stats==frozen and held_out.input_transform==stats
    with np.load(val.path,allow_pickle=False) as z:
        raw=z['X_raw'][:8]
    expected=encode_raw_imag(raw[:,2],stats)
    assert np.array_equal(expected,held_out.X[:,2].numpy())
    report['train_only_scaling_targets_unused_and_base_arrays_unchanged']=True
    if smoke_dir:
        with np.load(root/'dataset_test.npz',allow_pickle=False) as z:
            raw=z['X_raw'][:8]
            inputs={'X':z['X'][:8],'X_raw':raw,'schema_version':z['schema_version'],**{k:z[k] for k in STAT_KEYS}}
        for arch in ['axial','tri']:
            folder=Path(smoke_dir)/arch/'ddc_imag_pinv/seed_42';checkpoint=folder/'best.pt'
            p=predict_raw(checkpoint,raw[:,0],raw[:,2],device='cpu',batch_size=2)
            with np.load(folder/'test_predictions.npz') as z:assert np.allclose(p,z['prediction'],rtol=1e-6,atol=2e-6)
            pp=predict_raw(checkpoint,raw[:,0],raw[:,1]+1j*raw[:,2],device='cpu',batch_size=2)
            assert np.array_equal(p,pp)
            with tempfile.TemporaryDirectory(prefix='pinv_inference_',dir=Path(__file__).parent) as tmp:
                path=Path(tmp)/'inputs_only.npz';np.savez_compressed(path,**inputs)
                npz_p=predict_npz(checkpoint,path,device='cpu',batch_size=2)
                assert np.array_equal(p,npz_p)
        report['raw_component_complex_Yb_and_label_free_NPZ_inference_match_checkpoints']=True
    report['cuda_tested_locally']=False
    (Path(__file__).parent/'pinv_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',required=True);p.add_argument('--smoke-dir')
    a=p.parse_args();run(a.data_dir,a.smoke_dir)
