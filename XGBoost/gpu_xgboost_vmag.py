"""Four input ablations for independent-output XGBoost, DQ or Vmag targets.

Uses normalized v2 NPZ data. Sequential outputs reuse training-quantized
features; stable UBJ checkpoints support output-by-output recovery.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import numpy as np
import xgboost as xgb
from dataset import VARIANTS,load,features,check_splits,decode
from metrics import compute_metrics


class Heartbeat:
    def __init__(self,interval):self.interval=interval;self.stop_event=threading.Event();self.thread=None
    def __enter__(self):
        if self.interval>0:
            self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
        return self
    def _run(self):
        while not self.stop_event.wait(self.interval):print('[heartbeat] XGBoost training is active',flush=True)
    def __exit__(self,*args):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=1)


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def atomic_json(value,path):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def validate_device(device,threads):
    """Do not silently report a CUDA experiment that actually ran on CPU."""
    if re.fullmatch(r'cpu|cuda(?::\d+)?',device) is None:raise ValueError('device must be cpu, cuda, or cuda:N')
    if device=='cpu':return
    if not xgb.build_info().get('USE_CUDA',False):raise RuntimeError('This XGBoost build lacks CUDA support')
    d=xgb.DMatrix(np.arange(12,dtype=np.float32).reshape(6,2),label=np.arange(6,dtype=np.float32),nthread=threads)
    model=xgb.train({'tree_method':'hist','device':device,'max_depth':1,'nthread':threads},d,num_boost_round=1)
    actual=json.loads(model.save_config())['learner']['generic_param']['device']
    if not actual.startswith('cuda'):raise RuntimeError('CUDA requested but XGBoost fell back to CPU. Check GPU/build/driver, or explicitly use --device cpu')


def output_indices(count,smoke):
    return np.unique(np.linspace(0,count-1,min(12,count),dtype=np.int64)) if smoke else np.arange(count,dtype=np.int64)


def make_metadata(a,variant,seed,tr,indices):
    return {'schema_version':'ddc_yb_train_normalized_v2','variant':variant,'channels':VARIANTS[variant],
            'seed':seed,'target':a.target,'buses':tr['buses'],'normalization':tr['stats'],
            'normalization_signature':tr['signature'],'target_shape':tr['target_shape'],
            'output_indices':indices.tolist(),'complete_target':len(indices)==tr['y'].shape[1],
            'n_features':len(VARIANTS[variant])*tr['buses']**2,'data_fingerprints':a.data_fingerprints,
            'xgboost_version':xgb.__version__,'smoke':a.smoke,'checkpoint_format':'per_output_ubj_json_v1',
            'fit_settings':{k:getattr(a,k) for k in ['device','threads','rounds','max_depth','learning_rate','subsample','colsample_bytree','reg_alpha','reg_lambda','early_stopping','max_bin']}}


def checkpoint_booster(model,folder,idx):
    path=folder/f'output_{idx:05d}.ubj';tmp=folder/f'output_{idx:05d}.tmp.ubj'
    model.save_model(tmp);os.replace(tmp,path);return path


def load_state(folder):
    """Per-output records are authoritative even if the summary is interrupted."""
    folder=Path(folder);path=folder/'state.json'
    state=json.loads(path.read_text()) if path.exists() else {'outputs':{}}
    for record in sorted(folder.glob('record_*.json')):
        r=json.loads(record.read_text());state['outputs'][str(r['output_index'])]=r['info']
    return state


def train_trial(a,variant,seed,tr,va,indices):
    folder=Path(a.out_dir)/variant/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    meta=make_metadata(a,variant,seed,tr,indices);meta_path=folder/'metadata.json';state_path=folder/'state.json'
    if meta_path.exists():
        if not a.resume:raise FileExistsError(f'{folder}: use --resume or a fresh output directory')
        if json.loads(meta_path.read_text())!=meta:raise ValueError('Resume settings/data/version differ; use a fresh output directory')
    else:atomic_json(meta,meta_path)
    state=load_state(folder)
    for info in state['outputs'].values():
        if info['kind']=='booster':
            p=folder/info['file']
            if not p.exists() or file_hash(p)!=info['sha256']:raise ValueError('Missing/corrupt output checkpoint')
    print(f'\n{variant} seed={seed}: {meta["n_features"]} features, {len(indices)} outputs; {len(state["outputs"])} completed',flush=True)
    remaining=[int(i) for i in indices if str(int(i)) not in state['outputs']]
    if not remaining:return folder
    xt,xv=features(tr['X'],variant),features(va['X'],variant)
    # Validation quantiles reference only training data, never its own fitted cuts.
    dt=xgb.QuantileDMatrix(xt,max_bin=a.max_bin,nthread=a.threads)
    dv=xgb.QuantileDMatrix(xv,ref=dt,max_bin=a.max_bin,nthread=a.threads)
    params={'objective':'reg:squarederror','eval_metric':'rmse','tree_method':'hist','device':a.device,
            'nthread':a.threads,'max_depth':a.max_depth,'eta':a.learning_rate,'subsample':a.subsample,
            'colsample_bytree':a.colsample_bytree,'alpha':a.reg_alpha,'lambda':a.reg_lambda,'max_bin':a.max_bin}
    for idx in remaining:
        start=time.time();yt=tr['y'][:,idx];yv=va['y'][:,idx]
        if np.all(yt==yt[0]):info={'kind':'constant','value':float(yt[0]),'best_iteration':None,'seconds':0.}
        else:
            dt.set_label(yt);dv.set_label(yv)
            # Same output gets the same seed and initial training mean in every variant.
            config=dict(params,seed=int((seed+1009*idx)%(2**31-1)),base_score=float(yt.mean(dtype=float)))
            history={}
            model=xgb.train(config,dt,num_boost_round=a.rounds,evals=[(dv,'validation')],
                            early_stopping_rounds=a.early_stopping if a.early_stopping>0 else None,
                            evals_result=history,verbose_eval=False)
            rounds=len(history['validation']['rmse'])
            best=int(model.best_iteration) if a.early_stopping>0 else rounds-1
            # Native xgb.train otherwise returns trees beyond the best iteration.
            selected=model[:best+1]
            path=checkpoint_booster(selected,folder,idx)
            history_path=folder/f'history_{idx:05d}.json'
            atomic_json(history,history_path)
            info={'kind':'booster','file':path.name,'sha256':file_hash(path),'best_iteration':best,
                  'rounds_trained':rounds,'best_val_rmse_standardized':float(history['validation']['rmse'][best]),
                  'seconds':time.time()-start,'history_file':history_path.name}
        state['outputs'][str(idx)]=info
        atomic_json({'output_index':idx,'info':info},folder/f'record_{idx:05d}.json')
        done=len(state['outputs'])
        if done%a.progress_every==0 or done==len(indices):
            atomic_json(state,state_path)
            print(f'  completed {done}/{len(indices)} outputs',flush=True)
    return folder


def predict_normalized(folder,arrays,device='cpu',threads=4):
    """Predict a list of normalized full-slot X arrays, one loaded booster at a time."""
    folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text());state=load_state(folder)
    indices=meta['output_indices'];matrices=[];pred=[]
    for x in arrays:
        if x.ndim!=4 or x.shape[1:]!=(3,meta['buses'],meta['buses']):raise ValueError('Input geometry differs from trained model')
        matrices.append(xgb.DMatrix(features(x,meta['variant']),nthread=threads))
        pred.append(np.empty((len(x),len(indices)),dtype=np.float32))
    for j,idx in enumerate(indices):
        info=state['outputs'].get(str(idx))
        if info is None:raise ValueError(f'Output {idx} has no completed checkpoint; resume training first')
        if info['kind']=='constant':
            for p in pred:p[:,j]=info['value']
        else:
            model=xgb.Booster(params={'nthread':threads});model.load_model(folder/info['file'])
            model.set_param({'device':device,'nthread':threads})
            for p,d in zip(pred,matrices):p[:,j]=model.predict(d)
    return pred,meta


def evaluate(pred,split,meta,path=None):
    indices=np.asarray(meta['output_indices']);true_std=split['y'][:,indices]
    raw=decode(pred,meta['normalization'],meta['target'],indices);truth=split['raw'][:,indices]
    result=dict(compute_metrics(raw,truth),standardized=compute_metrics(pred,true_std))
    if meta['target']=='vmag':result['per_bus_r2']=[compute_metrics(raw[:,j],truth[:,j])['r2'] for j in range(raw.shape[1])]
    if meta['target']=='dq' and meta['complete_target']:
        n=meta['buses'];p=raw.reshape(len(raw),n,n);t=truth.reshape(len(truth),n,n)
        result['row_column_sum_metrics']=compute_metrics(np.concatenate([p.sum(1),p.sum(2)],1),np.concatenate([t.sum(1),t.sum(2)],1))
    if path is not None:
        shape=(len(raw),*meta['target_shape']) if meta['complete_target'] else raw.shape
        np.savez_compressed(path,prediction=raw.reshape(shape),target=truth.reshape(shape),
                            prediction_standardized=pred.reshape(shape),target_standardized=true_std.reshape(shape),
                            output_indices=indices,source_indices=split['ids'])
    return result


def aggregate(results,variants):
    report={}
    for variant in variants:
        rr=[r for r in results if r['variant']==variant];report[variant]={}
        for split in ['train','validation','test']:
            report[variant][split]={}
            for key in ['r2','rmse','mae','rmse_standardized','mae_standardized']:
                values=[r[split]['standardized'][key[:-13]] if key.endswith('_standardized') else r[split][key] for r in rr]
                values=[v for v in values if v is not None]
                report[variant][split][key]={'mean':float(np.mean(values)) if values else None,
                    'std_across_seeds':float(np.std(values,ddof=1)) if len(values)>1 else None}
    return report


def run(a):
    if int(xgb.__version__.split('.')[0])<2:raise RuntimeError('XGBoost >= 2.0 required')
    validate_device(a.device,a.threads)
    if a.smoke:
        a.rounds=5;a.max_depth=2;a.seeds=a.seeds[:1]
        print('SMOKE: train64/val32/test32, 12 selected outputs, five rounds. NOT a performance estimate.',flush=True)
    root=Path(a.data_dir);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    tr=load(root/'dataset_train.npz',a.target,64 if a.smoke else None)
    va=load(root/'dataset_val.npz',a.target,32 if a.smoke else None);check_splits(tr,va)
    if a.case is not None and a.case!=tr['buses']:raise ValueError('--case differs from dataset')
    a.data_fingerprints={s:file_hash(root/f'dataset_{s}.npz') for s in ['train','val']}
    indices=output_indices(tr['y'].shape[1],a.smoke)
    protocol={'arguments':vars(a),'normalization':tr['stats'],'normalization_signature':tr['signature'],
              'output_indices':indices.tolist(),'complete_target':len(indices)==tr['y'].shape[1],
              'test_used_for_selection':False,'primary_metrics':'original units','secondary_metrics':'standardized targets',
              'strategy':'one independent scalar booster per output; sequential fits; training-referenced quantile matrices'}
    atomic_json(protocol,out/'protocol.json')
    print('XGBoost',xgb.__version__,'device',a.device,'case',tr['buses'],'target',a.target,flush=True)
    results=[]
    for seed in a.seeds:
        for variant in a.variants:
            folder=train_trial(a,variant,seed,tr,va,indices)
            pp,meta=predict_normalized(folder,[tr['X'],va['X']],a.device,a.threads)
            r={'variant':variant,'seed':seed,'model_dir':str(folder.resolve()),'n_features':meta['n_features'],
               'n_outputs':len(indices),'complete_target':meta['complete_target'],
               'train':evaluate(pp[0],tr,meta),'validation':evaluate(pp[1],va,meta,folder/'validation_predictions.npz')}
            results.append(r);del pp
    print('\nAll validation-selected output models frozen. Loading test for final evaluation.',flush=True)
    te=load(root/'dataset_test.npz',a.target,32 if a.smoke else None);check_splits(tr,va,te)
    for r in results:
        folder=Path(r['model_dir']);pp,meta=predict_normalized(folder,[te['X']],a.device,a.threads)
        r['test']=evaluate(pp[0],te,meta,folder/'test_predictions.npz')
        atomic_json(r,folder/'results.json')
        print(r['variant'],r['seed'],'TEST raw',{k:r['test'][k] for k in ['r2','rmse','mae']},
              'standardized RMSE',r['test']['standardized']['rmse'],flush=True)
    summary={'protocol':protocol,'trials':results,'aggregate':aggregate(results,a.variants)}
    atomic_json(summary,out/'summary.json')
    rows=[{'variant':r['variant'],'seed':r['seed'],'n_features':r['n_features'],'n_outputs':r['n_outputs'],
           **{f'{s}_{k}':r[s][k] for s in ['train','validation','test'] for k in ['r2','rmse','mae']},
           **{f'{s}_{k}_standardized':r[s]['standardized'][k] for s in ['train','validation','test'] for k in ['r2','rmse','mae']}} for r in results]
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Saved',out.resolve(),flush=True)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True);p.add_argument('--out-dir',default='xgboost_ablation_results')
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    p.add_argument('--target','--target_key',type=lambda s:'dq' if s.lower() in ['y','dq'] else s.lower(),choices=['dq','vmag'],default='dq')
    p.add_argument('--case',type=int);p.add_argument('--seeds',nargs='+',type=int,default=[42,123,2026])
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--rounds','--n_estimators',type=int,default=600)
    p.add_argument('--max-depth','--max_depth',type=int,default=6)
    p.add_argument('--learning-rate','--learning_rate',type=float,default=.05)
    p.add_argument('--subsample',type=float,default=.8);p.add_argument('--colsample-bytree','--colsample_bytree',type=float,default=1.)
    p.add_argument('--reg-alpha','--reg_alpha',type=float,default=0.);p.add_argument('--reg-lambda','--reg_lambda',type=float,default=1.)
    p.add_argument('--early-stopping','--early_stopping_rounds',type=int,default=20)
    p.add_argument('--max-bin',type=int,default=256);p.add_argument('--progress-every',type=int,default=10)
    p.add_argument('--heartbeat-interval',type=int,default=120);p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if min(a.rounds,a.max_depth,a.threads,a.progress_every,a.max_bin)<1 or a.max_bin<2 or a.early_stopping<0 or a.learning_rate<=0 or a.reg_alpha<0 or a.reg_lambda<0:raise ValueError('Invalid numeric setting')
    if not 0<a.subsample<=1 or not 0<a.colsample_bytree<=1:raise ValueError('Sampling fractions must be in (0,1]')
    if len(set(a.variants))!=len(a.variants) or len(set(a.seeds))!=len(a.seeds):raise ValueError('Duplicate variants/seeds')
    return a


def main():
    a=cli()
    with Heartbeat(a.heartbeat_interval):run(a)


if __name__=='__main__':main()
