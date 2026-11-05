"""CPU verification on the supplied v2 datasets and generated smoke checkpoints."""
import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import train_transformer_imlive as training
from dataset import DependencyMatrixDataset,VARIANTS
from models import AblationModel,parameter_hash
from metrics import RunningMetrics,compute_metrics
from predict_ablation import predict_raw


def verify(data_root,code_root):
    torch.set_num_threads(4);report={}
    for buses in [30,39,57]:
        case=Path(data_root)/f'case{buses}'
        with np.load(case/'dataset_test.npz',allow_pickle=False) as z:
            x=z['X'][:8];raw=z['X_raw'][:8]
        for kind in ['axial','dq']:
            initial=[];params=[]
            for variant,channels in VARIANTS.items():
                torch.manual_seed(42)
                model=AblationModel(kind,variant,'dq',buses,16,1,4,.1,4.).eval()
                initial.append(parameter_hash(model));params.append(sum(p.numel() for p in model.parameters()))
                inp=torch.from_numpy(x[:2].copy()).requires_grad_(True)
                pred=model(inp)
                assert pred.shape==(2,1,buses,buses)
                pred.sum().backward()
                excluded=[c for c in range(3) if c not in channels]
                if excluded:
                    assert torch.count_nonzero(inp.grad[:,excluded])==0
                    changed=inp.detach().clone();changed[:,excluded]+=987
                    with torch.inference_mode():assert torch.equal(model(changed),model(inp.detach()))
                # Linear head permits negative outputs; internal attention gates stay.
                with torch.no_grad():
                    head=model.network.out.proj[-1];head.weight.zero_();head.bias.fill_(-2)
                with torch.inference_mode():assert (model(inp.detach())==-2).all()
                smoke=f'smoke_case{buses}'
                folder=Path(code_root)/smoke/kind/variant/'seed_42'
                pred_raw=predict_raw(folder/'best.pt',raw[:,0],raw[:,1]+1j*raw[:,2],device='cpu',batch_size=2)
                with np.load(folder/'test_predictions.npz') as z:
                    assert np.allclose(pred_raw,z['prediction'],rtol=1e-5,atol=2e-5)
            assert len(set(initial))==len(set(params))==1
        report[f'case{buses}']={'both_model_shapes_and_gradients':True,'excluded_input_values_and_gradients_blocked':True,
            'negative_outputs_supported':True,'paired_initial_weights_and_parameter_count':True,
            'raw_inference_matches_saved_test_predictions_all_four_variants':True}

    # Pooled metrics must be invariant to batch partitioning, including uneven batches.
    t=torch.linspace(-.2,1.4,37,dtype=torch.float64);p=t+.03*torch.sin(t*13)
    whole=compute_metrics(p,t);acc=RunningMetrics()
    for i in range(0,len(t),8):acc.update(p[i:i+8],t[i:i+8])
    assert all(abs(whole[k]-acc.compute()[k])<1e-12 for k in whole)
    report['streaming_metrics_match_global_uneven_batch_metrics']=True

    # Interrupt immediately after the first latest checkpoint, then resume epoch 2.
    case=Path(data_root)/'case30'
    tr=DependencyMatrixDataset(case/'dataset_train.npz','dq',16)
    va=DependencyMatrixDataset(case/'dataset_val.npz','dq',8)
    resume_workspace=tempfile.TemporaryDirectory(prefix='resume_check_',dir=code_root)
    folder=Path(resume_workspace.name)
    a=SimpleNamespace(epochs=2,batch_size=2,lr=3e-4,weight_decay=1e-4,loss_alpha=.5,
        patience=30,warmup_epochs=5,amp=False,deterministic=False,smoke=True,
        num_workers=0,d_model=16,depth=1,num_heads=4,dropout=.1,mlp_ratio=4.,target='dq',device=torch.device('cpu'),
        out_dir=str(folder),resume=False,
        data_fingerprints={s:training.file_hash(case/f'dataset_{s}.npz') for s in ['train','val']})
    save=training.atomic_save
    class Interrupted(Exception):pass
    def interrupt_after_first_epoch(value,path):
        save(value,path)
        if Path(path).name=='latest.pt' and value['epoch']==1:raise Interrupted()
    training.atomic_save=interrupt_after_first_epoch
    try:
        training.fit(a,'axial','ddc',42,tr,va)
    except Interrupted:pass
    finally:training.atomic_save=save
    a.resume=True;training.fit(a,'axial','ddc',42,tr,va)
    resumed=torch.load(folder/'axial/ddc/seed_42/latest.pt',map_location='cpu',weights_only=False)
    original=torch.load(Path(code_root)/'smoke_case30/axial/ddc/seed_42/latest.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(resumed['state'][k],v) for k,v in original['state'].items())
    assert folder.resolve().is_relative_to(Path(code_root).resolve())
    resume_workspace.cleanup()
    report['interrupted_epoch_resume_matches_uninterrupted_weights_exactly_cpu']=True
    report['cuda_tested_locally']=False
    report['note']='Small two-epoch CPU smoke tests establish compatibility, not predictive accuracy.'
    out=Path(code_root)/'verification.json';out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--code-root',default=str(Path(__file__).parent))
    a=p.parse_args();verify(a.data_root,a.code_root)
