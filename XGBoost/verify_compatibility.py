"""CPU acceptance checks, including complete synthetic matrix/vector runs."""
import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from dataset import VARIANTS,STAT_KEYS,features
from normalization import fitted_stats,encode_x
import gpu_xgboost_vmag as training
from predict_ablation import predict_raw,predict_npz


def synthetic_files(root):
    rng=np.random.default_rng(11);parts={}
    for name,n,start in [('train',24,0),('val',8,100),('test',8,200)]:
        x=rng.normal(size=(n,3,3,3));x[:,0]=np.abs(x[:,0])*3
        y=(x[:,0:1]*.7+np.abs(x[:,1:2])*.2);y[:,0,0,0]=2.
        v=.98+.01*x[:,0].mean(2)
        parts[name]=(x,y,v,np.arange(start,start+n))
    stats=fitted_stats(*parts['train'])
    for name,(x,y,v,ids) in parts.items():
        arrays={'X':encode_x(x,stats),'y':((y-stats['dq_center'])/stats['dq_scale']).astype(np.float32),
                'Vmag':((v-np.array(stats['vmag_center']))/np.array(stats['vmag_scale'])).astype(np.float32),
                'X_raw':x,'DQ_raw':y,'Vmag_raw':v,'source_indices':ids,
                'schema_version':np.array('ddc_yb_train_normalized_v2')}
        arrays.update({k:np.asarray(stats[k]) for k in STAT_KEYS})
        np.savez_compressed(root/f'dataset_{name}.npz',**arrays)


def settings(data,out,target='dq'):
    return SimpleNamespace(data_dir=str(data),out_dir=str(out),variants=list(VARIANTS),seeds=[42],target=target,
        case=3,device='cpu',threads=2,rounds=8,max_depth=2,learning_rate=.1,subsample=.8,
        colsample_bytree=1.,reg_alpha=0.,reg_lambda=1.,early_stopping=2,max_bin=16,
        progress_every=100,resume=False,smoke=False)


def verify(data_root,code_root):
    report={}
    for case in [30,39,57]:
        with np.load(Path(data_root)/f'case{case}'/'dataset_test.npz',allow_pickle=False) as z:
            x=z['X'][:32];raw=z['X_raw'][:32]
            inference={'X':x,'schema_version':z['schema_version'],**{k:z[k] for k in STAT_KEYS}}
        for variant,selected in VARIANTS.items():
            folder=Path(code_root)/f'smoke_case{case}'/variant/'seed_42'
            actual=predict_raw(folder,raw[:,0],raw[:,1]+1j*raw[:,2],device='cpu',threads=2)
            with np.load(folder/'test_predictions.npz') as z:assert np.allclose(actual,z['prediction'],rtol=1e-7,atol=1e-7)
            excluded=[i for i in range(3) if i not in selected]
            if excluded:
                changed=x.copy();changed[:,excluded]+=1234
                assert np.array_equal(features(changed,variant),features(x,variant))
        with tempfile.TemporaryDirectory(prefix='xgb_inference_',dir=code_root) as tmp:
            path=Path(tmp)/'input_only.npz';np.savez_compressed(path,**inference)
            actual=predict_npz(Path(code_root)/f'smoke_case{case}'/'ddc_yb/seed_42',path,device='cpu',threads=2)
            with np.load(Path(code_root)/f'smoke_case{case}'/'ddc_yb/seed_42/test_predictions.npz') as z:assert np.array_equal(actual,z['prediction'])
        report[f'case{case}']={'four_ablation_smoke_training':True,'raw_inference_matches_saved_predictions':True,
                             'excluded_feature_invariance':True,'inference_without_targets':True}
    with tempfile.TemporaryDirectory(prefix='xgb_full_test_',dir=code_root) as tmp:
        root=Path(tmp);data=root/'data';data.mkdir();synthetic_files(data)
        reference=root/'reference';training.run(settings(data,reference))
        resume=root/'resume';a=settings(data,resume)
        save=training.checkpoint_booster
        class Interrupted(Exception):pass
        def stop(model,folder,idx):
            result=save(model,folder,idx)
            if idx==3:raise Interrupted()
            return result
        training.checkpoint_booster=stop
        try:training.run(a)
        except Interrupted:pass
        finally:training.checkpoint_booster=save
        a.resume=True;training.run(a)
        for variant in VARIANTS:
            one=reference/variant/'seed_42';two=resume/variant/'seed_42'
            with np.load(one/'test_predictions.npz') as z,np.load(two/'test_predictions.npz') as zz:
                assert np.array_equal(z['prediction'],zz['prediction'])
                assert z['prediction'].shape==(8,1,3,3)
            state=json.loads((two/'state.json').read_text())
            assert state['outputs']['0']['kind']=='constant'
            import xgboost as xgb
            for info in state['outputs'].values():
                if info['kind']=='booster':
                    model=xgb.Booster();model.load_model(two/info['file'])
                    assert model.num_boosted_rounds()==info['best_iteration']+1
        training.run(settings(data,root/'vmag',target='vmag'))
        with np.load(root/'vmag/ddc_yb/seed_42/test_predictions.npz') as z:assert z['prediction'].shape==(8,3)
        report['complete_synthetic_matrix_and_vector_shapes']=True
        report['interrupted_output_resume_matches_uninterrupted_predictions_exactly']=True
        report['constant_outputs_and_best_iteration_slicing']=True
    report['cuda_tested_locally']=False
    report['note']='Real-data smoke checks use subsets and 12 selected outputs, not full predictive benchmarks.'
    (Path(code_root)/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--code-root',default=str(Path(__file__).parent))
    a=p.parse_args();verify(a.data_root,a.code_root)
