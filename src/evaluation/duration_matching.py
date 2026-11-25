"""Duration-stratified subsampling for analysis only; never changes official splits."""
import numpy as np

def duration_match_indices(simple_durations,compound_durations,bins=8,seed=42):
    s=np.asarray(simple_durations); c=np.asarray(compound_durations)
    if len(s)==0 or len(c)==0: return np.array([],int),np.array([],int),{'reason':'empty bucket'}
    edges=np.unique(np.quantile(np.concatenate([s,c]),np.linspace(0,1,bins+1)))
    if len(edges)<2: edges=np.array([float(np.min(s))-1e-6,float(np.max(s))+1e-6])
    rng=np.random.default_rng(seed); si=[]; ci=[]; report=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        sm=np.where((s>=lo)&(s<hi if hi<edges[-1] else s<=hi))[0]
        cm=np.where((c>=lo)&(c<hi if hi<edges[-1] else c<=hi))[0]
        n=min(len(sm),len(cm))
        if n:
            si.extend(rng.choice(sm,n,replace=False)); ci.extend(rng.choice(cm,n,replace=False))
        report.append({'lower':float(lo),'upper':float(hi),'simple_before':len(sm),'compound_before':len(cm),'matched_each':n})
    return np.asarray(si,int),np.asarray(ci,int),{'bins':report,'simple_after':len(si),'compound_after':len(ci)}
