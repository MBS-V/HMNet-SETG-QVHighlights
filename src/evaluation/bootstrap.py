import numpy as np

def bootstrap_ci(values,statistic=np.mean,n_resamples=1000,confidence=0.95,seed=42):
    x=np.asarray(values,dtype=float)
    if x.size==0: raise ValueError('Cannot bootstrap an empty sample')
    rng=np.random.default_rng(seed); samples=np.empty(n_resamples)
    for i in range(n_resamples): samples[i]=statistic(rng.choice(x,size=x.size,replace=True))
    alpha=(1-confidence)/2
    return {'estimate':float(statistic(x)),'lower':float(np.quantile(samples,alpha)),'upper':float(np.quantile(samples,1-alpha)),
            'confidence':confidence,'n_resamples':n_resamples,'n':int(x.size)}

def bootstrap_difference_in_differences(hm_simple,setg_simple,hm_compound,setg_compound,n_resamples=1000,confidence=0.95,seed=42):
    """Query-level paired bootstrap; each model pair is resampled on the same queries."""
    hs=np.asarray(hm_simple,dtype=float); ss=np.asarray(setg_simple,dtype=float)
    hc=np.asarray(hm_compound,dtype=float); sc=np.asarray(setg_compound,dtype=float)
    if len(hs)!=len(ss) or len(hc)!=len(sc) or not len(hs) or not len(hc):
        raise ValueError('Simple and compound arrays must be non-empty and paired by query within each group')
    estimate=float(np.mean(sc-hc)-np.mean(ss-hs)); rng=np.random.default_rng(seed); draws=np.empty(n_resamples)
    for i in range(n_resamples):
        si=rng.integers(0,len(hs),len(hs)); ci=rng.integers(0,len(hc),len(hc))
        draws[i]=np.mean(sc[ci]-hc[ci])-np.mean(ss[si]-hs[si])
    alpha=(1-confidence)/2; lo=float(np.quantile(draws,alpha)); hi=float(np.quantile(draws,1-alpha))
    return {'estimate':estimate,'lower':lo,'upper':hi,'confidence':confidence,'n_resamples':n_resamples,
            'simple_queries':len(hs),'compound_queries':len(hc),'includes_zero':bool(lo<=0<=hi)}
