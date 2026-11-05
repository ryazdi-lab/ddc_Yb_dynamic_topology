"""Pooled regression metrics, accumulated as sums rather than batch R2 averages."""
import math
import torch
from torch import nn


class CombinedMSEMAELoss(nn.Module):
    def __init__(self,alpha=0.5):
        super().__init__()
        if not 0<=alpha<=1:raise ValueError('alpha must be in [0,1]')
        self.alpha=alpha
    def forward(self,pred,target):
        if pred.shape!=target.shape:raise ValueError('Prediction/target shape mismatch')
        error=pred.float()-target.float()
        return self.alpha*error.square().mean()+(1-self.alpha)*error.abs().mean()


class RunningMetrics:
    def __init__(self):self.n=0;self.sse=0.;self.sae=0.;self.mean=0.;self.m2=0.
    def update(self,pred,target):
        if pred.shape!=target.shape:raise ValueError('Prediction/target shape mismatch')
        p,t=pred.detach().double(),target.detach().double()
        if not torch.isfinite(p).all() or not torch.isfinite(t).all():raise FloatingPointError('Nonfinite regression values')
        error=p-t;count=t.numel();mean=float(t.mean());m2=float(((t-mean)**2).sum())
        self.sse+=float(error.square().sum());self.sae+=float(error.abs().sum())
        delta=mean-self.mean;total=self.n+count
        self.m2+=m2+delta*delta*self.n*count/total
        self.mean+=delta*count/total;self.n=total
    def compute(self):
        if self.n==0:raise ValueError('Empty evaluation')
        return {'r2':1-self.sse/self.m2 if self.m2>1e-20 else None,
                'rmse':math.sqrt(self.sse/self.n),'mae':self.sae/self.n}


def compute_metrics(pred,target):
    acc=RunningMetrics();acc.update(pred,target);return acc.compute()
