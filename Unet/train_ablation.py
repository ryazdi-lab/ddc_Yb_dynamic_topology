"""GPU training for the four matched Ddc/Yb input ablations.

Requires the dl_normalized_datasets v2 files, NumPy and PyTorch >= 2.5.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import DependencyMatrixDataset,VARIANTS,check_splits
from metrics import CombinedMSEMAELoss,RunningMetrics,compute_metrics
from models import AblationModel,parameter_hash


def seed_all(seed,deterministic=False):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark=not deterministic


def device_for(name):
    if name=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; install CUDA-enabled PyTorch or use --device cpu')
    return torch.device('cuda' if name=='cuda' or (name=='auto' and torch.cuda.is_available()) else 'cpu')


def atomic_save(value,path):
    tmp=Path(str(path)+'.tmp');torch.save(value,tmp);os.replace(tmp,path)


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_history(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def loader_for(ds,args,shuffle=False,generator=None):
    return DataLoader(ds,batch_size=args.batch_size,shuffle=shuffle,generator=generator,
                      num_workers=args.num_workers,pin_memory=args.device.type=='cuda',
                      persistent_workers=args.num_workers>0)


def train_epoch(model,loader,criterion,opt,scaler,device,amp):
    model.train();total=0.;count=0;acc=RunningMetrics()
    for x,y,raw in loader:
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
            pred=model(x);loss=criterion(pred,y)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; try --no-amp or a lower learning rate')
        scaler.scale(loss).backward();scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(opt);scaler.update()
        physical=loader.dataset.denormalize_y(pred.detach().float().cpu().double())
        acc.update(physical,raw)
        total+=float(loss.detach())*len(x);count+=len(x)
    return dict(acc.compute(),loss=total/count)


@torch.inference_mode()
def evaluate(model,loader,criterion,device,amp,save_path=None):
    model.eval();total=0.;count=0;acc=RunningMetrics();pp=[];tt=[]
    for x,y,raw in loader:
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
            pred=model(x.to(device,non_blocking=True))
            loss=criterion(pred,y.to(device,non_blocking=True))
        physical=loader.dataset.denormalize_y(pred.float().cpu().double())
        acc.update(physical,raw);total+=float(loss)*len(x);count+=len(x)
        if save_path is not None:pp.append(physical.numpy());tt.append(raw.numpy())
    result=dict(acc.compute(),loss=total/count)
    if save_path is not None:
        p,t=np.concatenate(pp),np.concatenate(tt)
        np.savez_compressed(save_path,prediction=p,target=t,source_indices=loader.dataset.ids)
        if loader.dataset.target_key=='vmag':
            result['per_bus']=[compute_metrics(torch.from_numpy(p[:,i]),torch.from_numpy(t[:,i])) for i in range(p.shape[1])]
    return result


def make_metadata(args,model_type,variant,seed,train,initial_hash):
    fit_settings={k:getattr(args,k) for k in ['epochs','batch_size','lr','weight_decay','loss_alpha','patience','lr_patience','lr_factor','amp','deterministic','smoke','num_workers']}
    return {'model_config':{'model_type':model_type,'variant':variant,'target':args.target,
                           'base_channels':args.base_channels,'attn_heads':args.attn_heads},
            'normalization':train.stats_dict(),'normalization_signature':train.signature,
            'buses':train.buses,'seed':seed,'initial_parameter_hash':initial_hash,
            'data_fingerprints':args.data_fingerprints,
            'training_source_indices_hash':hashlib.sha256(train.ids.astype('<i8').tobytes()).hexdigest(),
            'fit_settings':fit_settings,'selection':'lowest raw-unit pooled validation RMSE',
            'schema_version':'ddc_yb_train_normalized_v2'}


def fit(args,model_type,variant,seed,train,val):
    seed_all(seed,args.deterministic)
    folder=Path(args.out_dir)/model_type/variant/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    if not args.resume and any((folder/name).exists() for name in ['best.pt','latest.pt']):
        raise FileExistsError(f'{folder}: use --resume or a fresh --out-dir')
    model=AblationModel(model_type,variant,args.target,args.base_channels,args.attn_heads).to(args.device)
    initial_hash=parameter_hash(model);meta=make_metadata(args,model_type,variant,seed,train,initial_hash)
    generator=torch.Generator().manual_seed(seed)
    tr=loader_for(train,args,shuffle=True,generator=generator);va=loader_for(val,args)
    criterion=CombinedMSEMAELoss(args.loss_alpha)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',patience=args.lr_patience,factor=args.lr_factor)
    amp=args.amp and args.device.type=='cuda'
    scaler=torch.amp.GradScaler(args.device.type,enabled=amp)
    start=1;best=float('inf');stale=0;rows=[]
    if args.resume and (folder/'latest.pt').exists():
        s=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        if s['metadata']!=meta:raise ValueError('Resume data/statistics/settings differ; use a fresh output directory')
        model.load_state_dict(s['state']);opt.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler']);scaler.load_state_dict(s['scaler'])
        start=s['epoch']+1;best=s['best'];stale=s['stale'];rows=s['history']
        generator.set_state(s['loader_rng']);torch.set_rng_state(s['torch_rng'])
        random.setstate(s['python_rng']);np.random.set_state(s['numpy_rng'])
        if args.device.type=='cuda' and s['cuda_rng'] is not None:torch.cuda.set_rng_state_all(s['cuda_rng'])
    parameters=sum(p.numel() for p in model.parameters())
    print(f'\n{model_type} {variant} seed={seed}: {parameters:,} parameters; AMP={amp}',flush=True)
    for epoch in range(start,args.epochs+1):
        if stale>=args.patience:break
        tick=time.time();tm=train_epoch(model,tr,criterion,opt,scaler,args.device,amp)
        vm=evaluate(model,va,criterion,args.device,amp)
        row={'epoch':epoch,'lr':opt.param_groups[0]['lr'],
             **{f'train_{k}':v for k,v in tm.items()},**{f'val_{k}':v for k,v in vm.items()},'seconds':time.time()-tick}
        rows.append(row);scheduler.step(vm['rmse'])
        if vm['rmse']<best:
            best=vm['rmse'];stale=0
            atomic_save({'state':model.state_dict(),'metadata':meta,'epoch':epoch,'validation':vm},folder/'best.pt')
        else:stale+=1
        atomic_save({'state':model.state_dict(),'metadata':meta,'epoch':epoch,'best':best,'stale':stale,'history':rows,
                     'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                     'torch_rng':torch.get_rng_state(),'loader_rng':generator.get_state(),'python_rng':random.getstate(),
                     'numpy_rng':np.random.get_state(),'cuda_rng':torch.cuda.get_rng_state_all() if args.device.type=='cuda' else None},folder/'latest.pt')
        write_history(folder/'history.csv',rows)
        print(f"epoch={epoch:3d} loss={tm['loss']:.5f} val_R2={vm['r2']} val_RMSE={vm['rmse']:.6f} ({row['seconds']:.1f}s)",flush=True)
    best_state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(best_state['state'])
    train_result=evaluate(model,loader_for(train,args),criterion,args.device,amp)
    val_result=evaluate(model,va,criterion,args.device,amp,folder/'validation_predictions.npz')
    result={'model':model_type,'variant':variant,'seed':seed,'parameters':parameters,
            'initial_parameter_hash':initial_hash,'best_epoch':best_state['epoch'],
            'checkpoint':str((folder/'best.pt').resolve()),'train':train_result,'validation':val_result}
    return result


def aggregate(results,args):
    report={}
    for model_type in args.models:
        report[model_type]={}
        for variant in args.variants:
            trials=[r for r in results if r['model']==model_type and r['variant']==variant]
            report[model_type][variant]={}
            for split in ['train','validation','test']:
                report[model_type][variant][split]={}
                for metric in ['r2','rmse','mae']:
                    values=[r[split][metric] for r in trials if r[split][metric] is not None]
                    report[model_type][variant][split][metric]={
                        'mean':float(np.mean(values)) if values else None,
                        'std_across_seeds':float(np.std(values,ddof=1)) if len(values)>1 else None}
    return report


def mean_baseline(ds,batch_size):
    acc=RunningMetrics()
    for i in range(0,len(ds),batch_size):
        truth=ds.raw[i:i+batch_size]
        prediction=ds.denormalize_y(torch.zeros_like(truth))
        acc.update(prediction,truth)
    return acc.compute()


def run(args):
    torch.set_num_threads(args.cpu_threads);args.device=device_for(args.device)
    if args.smoke:
        args.epochs=2;args.batch_size=4;args.base_channels=8;args.num_workers=0;args.seeds=args.seeds[:1]
        print('SMOKE: train16/val8/test8, two epochs. Scores are not performance estimates.',flush=True)
    root=Path(args.data_dir);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    print('PyTorch',torch.__version__,'device',args.device,flush=True)
    train=DependencyMatrixDataset(root/'dataset_train.npz',args.target,16 if args.smoke else None)
    val=DependencyMatrixDataset(root/'dataset_val.npz',args.target,8 if args.smoke else None)
    check_splits(train,val)
    args.data_fingerprints={split:file_hash(root/f'dataset_{split}.npz') for split in ['train','val']}
    print(train,'\n',val,flush=True)
    if args.base_channels<4 or (args.base_channels*8)%args.attn_heads:raise ValueError('Use base_channels>=4, with 8*base_channels divisible by attn_heads')
    protocol={'arguments':{k:str(v) if isinstance(v,torch.device) else v for k,v in vars(args).items()},
              'dataset_schema':'ddc_yb_train_normalized_v2','buses':train.buses,
              'normalization':train.stats_dict(),'normalization_signature':train.signature,
              'variants':VARIANTS,'fixed_input_slots':3,'mask_after_normalization':True,
              'metric_units':'raw DQ' if args.target=='dq' else 'voltage per unit',
              'loss_units':'standardized target','test_used_for_selection':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    results=[];paired={}
    for model_type in args.models:
        for seed in args.seeds:
            for variant in args.variants:
                r=fit(args,model_type,variant,seed,train,val)
                key=(model_type,seed)
                if key in paired and paired[key]!=r['initial_parameter_hash']:raise AssertionError('Paired initialization differs')
                paired[key]=r['initial_parameter_hash'];results.append(r)
    print('\nAll validation-selected checkpoints frozen. Loading test for final evaluation.',flush=True)
    test=DependencyMatrixDataset(root/'dataset_test.npz',args.target,8 if args.smoke else None)
    check_splits(train,val,test);te=loader_for(test,args)
    criterion=CombinedMSEMAELoss(args.loss_alpha);amp=args.amp and args.device.type=='cuda'
    for r in results:
        state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        model=AblationModel(**state['metadata']['model_config']).to(args.device);model.load_state_dict(state['state'])
        folder=Path(r['checkpoint']).parent
        r['test']=evaluate(model,te,criterion,args.device,amp,folder/'test_predictions.npz')
        (folder/'results.json').write_text(json.dumps(r,indent=2))
        print(r['model'],r['variant'],r['seed'],'TEST',{k:r['test'][k] for k in ['r2','rmse','mae']},flush=True)
    summary={'protocol':protocol,'trials':results,'aggregate':aggregate(results,args),
             'training_mean_baseline':{name:mean_baseline(ds,args.batch_size) for name,ds in [('train',train),('validation',val),('test',test)]}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    rows=[{'model':r['model'],'variant':r['variant'],'seed':r['seed'],'parameters':r['parameters'],'best_epoch':r['best_epoch'],
           **{f'{split}_{k}':r[split][k] for split in ['train','validation','test'] for k in ['r2','rmse','mae']}} for r in results]
    write_history(out/'comparison.csv',rows)
    print('Saved',out.resolve(),flush=True)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True);p.add_argument('--out-dir',default='unet_ablation_results')
    p.add_argument('--models',nargs='+',choices=['unet','attention_unet'],default=['unet'])
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    p.add_argument('--target',choices=['dq','vmag'],default='dq')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,123,2026])
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--epochs',type=int,default=200);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--base-channels',type=int,default=32);p.add_argument('--attn-heads',type=int,default=4)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--loss-alpha',type=float,default=.5,help='MSE weight; 1 means pure MSE')
    p.add_argument('--patience',type=int,default=30);p.add_argument('--lr-patience',type=int,default=8)
    p.add_argument('--lr-factor',type=float,default=.5)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--deterministic',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--cpu-threads',type=int,default=4)
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.patience,a.cpu_threads,a.attn_heads)<1 or a.num_workers<0 or a.lr<=0 or a.weight_decay<0 or a.lr_patience<0:raise ValueError('Invalid numeric setting')
    if not 0<=a.loss_alpha<=1 or not 0<a.lr_factor<1:raise ValueError('Invalid loss/scheduler setting')
    if any(len(set(v))!=len(v) for v in [a.models,a.variants,a.seeds]):raise ValueError('Models, variants and seeds must be unique')
    return a


if __name__=='__main__':run(cli())
