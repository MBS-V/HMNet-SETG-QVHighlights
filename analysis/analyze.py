import json, sys, numpy as np, collections
sys.path.insert(0,'/home/claude/ext/moment_detr')
from standalone_eval.eval import compute_average_precision_detection_wrapper, compute_mr_ap, compute_mr_r1
from standalone_eval.utils import load_jsonl
R='/tmp/res_in/results/'; OUT='/home/claude/final/'
RUNS=['hmnet_baseline_e50','hmnet_setg_e50','hmnet_baseline_fix','hmnet_setg_fix']
gt=load_jsonl('/home/claude/ext/moment_detr/data/highlight_val_release.jsonl'); G={d['qid']:d for d in gt}
tags={json.loads(l)['qid']:json.loads(l) for l in open('/home/claude/realrun/tags_val.jsonl')}
thr=[round(x,2) for x in np.linspace(0.5,0.95,10)]
def iou(a,b): i=max(0,min(a[1],b[1])-max(a[0],b[0])); u=max(a[1],b[1])-min(a[0],b[0]); return i/u if u>0 else 0
res={}
for r in RUNS:
    sub=load_jsonl(R+r+'/predictions_official.jsonl'); S={d['qid']:d for d in sub}
    qids=sorted(S)
    perq={}
    for q in qids:
        gtw=[{'video-id':q,'t-start':w[0],'t-end':w[1]} for w in G[q]['relevant_windows']]
        pr=[{'video-id':q,'t-start':w[0],'t-end':w[1],'score':w[2]} for w in S[q]['pred_relevant_windows'][:10]]
        perq[q]=compute_average_precision_detection_wrapper([q,gtw,pr],tiou_thresholds=thr)[1]
    top1={q:max(iou(S[q]['pred_relevant_windows'][0][:2],w) for w in G[q]['relevant_windows']) for q in qids}
    off=json.load(open(R+r+'/metrics_official.json'))['brief']
    ap=np.array([perq[q] for q in qids])
    res[r]=dict(qids=qids,ap=ap,top1=np.array([top1[q] for q in qids]),off=off,
        pred_end=np.array([S[q]['pred_relevant_windows'][0][1]>=G[q]['duration']-2 for q in qids]),
        pred_len=np.array([S[q]['pred_relevant_windows'][0][1]-S[q]['pred_relevant_windows'][0][0] for q in qids]),
        pred_start=np.array([S[q]['pred_relevant_windows'][0][0] for q in qids]))
    assert abs(100*ap.mean()-off['MR-full-mAP'])<0.02, (r,100*ap.mean(),off['MR-full-mAP'])
    assert abs(100*(res[r]['top1']>=0.5).mean()-off['MR-full-R1@0.5'])<0.02
qids=res[RUNS[0]]['qids']; assert all(res[r]['qids']==qids for r in RUNS)
comp=np.array([tags[q]['k']>=2 for q in qids]); print('compound n',comp.sum(),'simple n',(~comp).sum())
gt_end=np.array([any(w[1]>=G[q]['duration']-2 for w in G[q]['relevant_windows']) for q in qids])
gt_len=np.array([np.mean([w[1]-w[0] for w in G[q]['relevant_windows']]) for q in qids])
rng=np.random.default_rng(0); B=10000; idx=rng.integers(0,len(qids),(B,len(qids)))
def boot_diff(a,b,mask=None):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if mask is not None: a,b=a[mask],b[mask]
    n=len(a); ii=rng.integers(0,n,(B,n)); d=(b[ii]-a[ii]).mean(1)*100
    return round(100*(b.mean()-a.mean()),2), round(float(np.percentile(d,2.5)),2), round(float(np.percentile(d,97.5)),2)
def mcnemar(a,b,mask=None):
    from scipy.stats import binomtest
    a=np.asarray(a); b=np.asarray(b)
    if mask is not None: a,b=a[mask],b[mask]
    n01=int((~a&b).sum()); n10=int((a&~b).sum())
    p=binomtest(n01,n01+n10,0.5).pvalue if n01+n10>0 else 1.0
    return n10,n01,round(float(p),3)
summary={'runs':{},'comparisons':[],'bias':{},'n':{'all':len(qids),'compound':int(comp.sum()),'simple':int((~comp).sum())}}
for r in RUNS:
    d=res[r]; t=d['top1']; ap=d['ap']
    summary['runs'][r]={'best_epoch':None,'epochs':None,'R1@0.3':round(100*(t>=0.3).mean(),2),'R1@0.5':d['off']['MR-full-R1@0.5'],'R1@0.7':d['off']['MR-full-R1@0.7'],
      'mAP@0.5':d['off']['MR-full-mAP@0.5'],'mAP@0.75':d['off']['MR-full-mAP@0.75'],'mAP_avg':d['off']['MR-full-mAP'],'mIoU':round(100*t.mean(),2),
      'mAP_short':d['off']['MR-short-mAP'],'mAP_middle':d['off']['MR-middle-mAP'],'mAP_long':d['off']['MR-long-mAP'],
      'simple':{'R1@0.5':round(100*(t[~comp]>=0.5).mean(),2),'mAP_avg':round(100*ap[~comp].mean(),2)},
      'compound':{'R1@0.5':round(100*(t[comp]>=0.5).mean(),2),'mAP_avg':round(100*ap[comp].mean(),2)}}
    m=json.load(open(R+r+'/experiment_manifest.json')); h=m['history']
    summary['runs'][r]['epochs']=len(h); summary['runs'][r]['best_epoch']=int(max(h,key=lambda e:e['validation']['metrics']['mAP_avg'])['epoch'])
    summary['bias'][r]={'pred_ends_at_video_end_%':round(100*d['pred_end'].mean(),1),'mean_pred_len_s':round(float(d['pred_len'].mean()),1),
       'unique_top1_windows':int(len({(round(s,1),round(s+l,1)) for s,l in zip(d['pred_start'],d['pred_len'])}))}
summary['bias']['ground_truth']={'gt_has_window_at_video_end_%':round(100*gt_end.mean(),1),'mean_gt_len_s':round(float(gt_len.mean()),1)}
pairs=[('hmnet_baseline_e50','hmnet_setg_e50','SE-TG vs baseline (original settings)'),('hmnet_baseline_fix','hmnet_setg_fix','SE-TG vs baseline (corrected settings)'),
       ('hmnet_baseline_e50','hmnet_baseline_fix','corrected vs original (baseline)'),('hmnet_setg_e50','hmnet_setg_fix','corrected vs original (SE-TG)')]
for a,b,name in pairs:
    for sub,mask in [('all',None),('simple',~comp),('compound',comp)]:
        ha=res[a]['top1']>=0.5; hb=res[b]['top1']>=0.5
        n10,n01,p=mcnemar(ha,hb,mask)
        dR=boot_diff(ha,hb,mask); dM=boot_diff(res[a]['ap'].mean(1),res[b]['ap'].mean(1),mask)
        summary['comparisons'].append({'comparison':name,'A':a,'B':b,'subset':sub,'delta_R1@0.5':dR[0],'R1_CI95':[dR[1],dR[2]],
          'mcnemar_only_A_hits':n10,'mcnemar_only_B_hits':n01,'mcnemar_p':p,'delta_mAP_avg':dM[0],'mAP_CI95':[dM[1],dM[2]]})
json.dump(summary,open(OUT+'summary.json','w'),indent=1)
np.savez(OUT+'perquery.npz',**{r+'_ap':res[r]['ap'] for r in RUNS},**{r+'_top1':res[r]['top1'] for r in RUNS},comp=comp,qids=np.array(qids))
print(json.dumps(summary['runs'],indent=0)[:2500]); print(json.dumps(summary['bias'],indent=0))
for c in summary['comparisons']: print(c['comparison'][:40],c['subset'],'dR1',c['delta_R1@0.5'],c['R1_CI95'],'p',c['mcnemar_p'],'| dmAP',c['delta_mAP_avg'],c['mAP_CI95'])
