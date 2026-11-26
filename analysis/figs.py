import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948']; GREY='#9a9a9a'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#888888','axes.grid':True,'grid.color':'#e6e6e6','axes.axisbelow':True})
R='/tmp/res_in/results/'; O='/home/claude/final/figures/'; import os; os.makedirs(O,exist_ok=True)
S=json.load(open('/home/claude/final/summary.json'))
RUNS=['hmnet_baseline_e50','hmnet_setg_e50','hmnet_baseline_fix','hmnet_setg_fix']
LAB={'hmnet_baseline_e50':'HM-Net (original)','hmnet_setg_e50':'HM-Net + SE-TG (original)','hmnet_baseline_fix':'HM-Net (corrected)','hmnet_setg_fix':'HM-Net + SE-TG (corrected)'}
COL={'hmnet_baseline_e50':C[0],'hmnet_setg_e50':C[1],'hmnet_baseline_fix':C[2],'hmnet_setg_fix':C[6]}
LS={'hmnet_baseline_e50':'-','hmnet_setg_e50':'-','hmnet_baseline_fix':'--','hmnet_setg_fix':'--'}
SRC='Source: our runs, official Moment-DETR evaluator, QVHighlights val (1,550 queries)'
FLOOR={'R1@0.5':9.03,'R1@0.7':6.06,'mAP@0.5':17.58,'mAP@0.75':9.87,'mAP_avg':10.85}
def save(n,src=SRC):
    plt.gcf().text(0.01,0.005,src,fontsize=7,color='#666666'); plt.tight_layout(rect=(0,0.03,1,1)); plt.savefig(O+n,dpi=180); plt.close()
H={r:json.load(open(R+r+'/experiment_manifest.json'))['history'] for r in RUNS}
# R1 main results
M=['R1@0.5','R1@0.7','mAP@0.5','mAP@0.75','mAP_avg']; x=np.arange(len(M)); w=0.16
fig,ax=plt.subplots(figsize=(10.5,4.4))
for i,r in enumerate(RUNS):
    v=[S['runs'][r][m] for m in M]; ax.bar(x+(i-2)*w,v,w-0.02,color=COL[r],label=LAB[r],hatch=None if 'e50' in r else '//',edgecolor='white',linewidth=0.5)
    for j,y in enumerate(v): ax.text(x[j]+(i-2)*w,y+0.5,f'{y:.1f}',ha='center',fontsize=6.5,color='#333333')
ax.bar(x+2*w,[FLOOR[m] for m in M],w-0.02,color=GREY,label='Query-blind prior (measured)')
ax.set_xticks(x,[m.replace('_avg',' avg') for m in M]); ax.set_ylabel('%'); ax.set_ylim(0,34)
ax.legend(frameon=False,ncol=3,fontsize=8,loc='upper left'); ax.set_title('Main results on QVHighlights val (best checkpoint per run). Published HM-Net: R1@0.5 62.8, mAP avg 43.2',fontsize=10)
save('R1_main_results.png')
# R2 / R4 curves
for key,fn,title,floor in [('mAP_avg','R2_val_mAP_avg_per_epoch.png','Validation mAP avg per epoch (internal metric)',10.85),('R1@0.5','R4_val_R1_per_epoch.png','Validation R1@0.5 per epoch (internal metric)',9.03)]:
    fig,ax=plt.subplots(figsize=(8.5,4.2))
    for r in RUNS:
        e=[h['epoch'] for h in H[r]]; v=[h['validation']['metrics'][key] for h in H[r]]
        ax.plot(e,v,LS[r],color=COL[r],lw=2,label=LAB[r]); b=int(np.argmax([h['validation']['metrics']['mAP_avg'] for h in H[r]]))
        ax.plot(e[b],v[b],'o',ms=8,color=COL[r],markeredgecolor='white',markeredgewidth=1.5)
    ax.axhline(floor,color=GREY,ls=':',lw=1.5); ax.text(29.5,floor+0.25,'query-blind prior',ha='right',fontsize=8,color='#555555')
    ax.axvspan(-0.5,4.5,color='#f3f3f3',zorder=0); ax.text(2,ax.get_ylim()[1]*0.97,'warm-up',ha='center',fontsize=8,color='#777777',va='top')
    ax.set_xlabel('epoch'); ax.set_ylabel('%'); ax.set_xlim(-0.5,29.5); ax.legend(frameon=False,fontsize=8,ncol=2,loc='lower right'); ax.set_title(title+'; dots = selected best checkpoint',fontsize=10)
    save(fn,'Source: our training logs (experiment_manifest.json); internal validation metrics on the 1,550 val queries')
# R3 train loss: two panels
fig,axs=plt.subplots(1,2,figsize=(10,3.8),sharex=True)
for ax,grp,t in [(axs[0],['hmnet_baseline_e50','hmnet_setg_e50'],'Original settings'),(axs[1],['hmnet_baseline_fix','hmnet_setg_fix'],'Corrected settings (L2 norm + background weight)')]:
    for r in grp:
        e=[h['epoch'] for h in H[r]]
        ax.plot(e,[h['train']['loss'] for h in H[r]],'-',color=COL[r],lw=2,label=LAB[r]+' train')
        ax.plot(e,[h['validation']['loss'] for h in H[r]],':',color=COL[r],lw=2,label=LAB[r]+' val')
    ax.set_title(t,fontsize=10); ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.legend(frameon=False,fontsize=7)
fig.suptitle('Training vs validation loss: training keeps falling, validation flattens (loss scales differ between panels)',fontsize=10)
save('R3_loss_per_epoch.png','Source: our training logs (experiment_manifest.json)')
# R6 forest plot SE-TG vs baseline
fig,axs=plt.subplots(1,2,figsize=(10,3.9),sharey=True)
rows=[c for c in S['comparisons'] if c['comparison'].startswith('SE-TG vs baseline')]
labels=[]; 
for k,(ax,dk,ck,t) in enumerate([(axs[0],'delta_R1@0.5','R1_CI95','Δ R1@0.5 (SE-TG − baseline)'),(axs[1],'delta_mAP_avg','mAP_CI95','Δ mAP avg (SE-TG − baseline)')]):
    for i,c in enumerate(rows):
        col=C[0] if 'original' in c['comparison'] else C[6]
        ax.plot(c[ck],[i,i],color=col,lw=2.5); ax.plot(c[dk],i,'o',color=col,ms=7,markeredgecolor='white')
        ax.text(c[ck][1]+0.25,i,f"{c[dk]:+.1f}" + (f"  p={c['mcnemar_p']:.2f}" if k==0 else ''),va='center',fontsize=7.5,color='#333333')
    ax.axvline(0,color='#555555',lw=1); ax.set_title(t,fontsize=10); ax.set_xlabel('percentage points (95% bootstrap CI)')
    ax.grid(axis='y',visible=False)
axs[0].set_yticks(range(len(rows)),[('original' if 'original' in c['comparison'] else 'corrected')+f" · {c['subset']} (n={S['n'][c['subset']]})" for c in rows]); axs[0].invert_yaxis()
fig.suptitle('Does SE-TG help? No 95% interval excludes 0 (closest: corrected·simple, p = 0.06) → fail to reject H0',fontsize=10.5)
save('R6_setg_vs_baseline_paired.png','Source: our runs; paired bootstrap (10,000 resamples) over the same val queries; p = exact McNemar test on top-1 hits (IoU≥0.5)')
# R5 original vs corrected
fig,axs=plt.subplots(1,2,figsize=(9,3.6))
for ax,m in zip(axs,['mAP_avg','R1@0.5']):
    for i,(a,b,n) in enumerate([('hmnet_baseline_e50','hmnet_baseline_fix','HM-Net'),('hmnet_setg_e50','hmnet_setg_fix','HM-Net + SE-TG')]):
        ax.bar(i-0.18,S['runs'][a][m],0.34,color=C[0],label='original' if i==0 else None); ax.bar(i+0.18,S['runs'][b][m],0.34,color=C[2],hatch='//',edgecolor='white',label='corrected' if i==0 else None)
        ax.text(i-0.18,S['runs'][a][m]+0.3,f"{S['runs'][a][m]:.1f}",ha='center',fontsize=8); ax.text(i+0.18,S['runs'][b][m]+0.3,f"{S['runs'][b][m]:.1f}",ha='center',fontsize=8)
    ax.axhline(FLOOR[m],color=GREY,ls=':'); ax.text(1.45,FLOOR[m]+0.3,'query-blind',fontsize=7,ha='right',color='#555555')
    ax.set_xticks([0,1],['HM-Net','HM-Net + SE-TG']); ax.set_title(m.replace('_avg',' avg'),fontsize=10); ax.set_ylabel('%'); ax.set_ylim(0,20); ax.legend(frameon=False,fontsize=8)
fig.suptitle('Original vs corrected settings: no significant change (all paired CIs cross 0)',fontsize=10)
save('R5_original_vs_corrected.png')
# R8 end bias + R9 by length
fig,axs=plt.subplots(1,2,figsize=(10.5,3.8))
names=['ground_truth']+RUNS; vals=[S['bias']['ground_truth']['gt_has_window_at_video_end_%']]+[S['bias'][r]['pred_ends_at_video_end_%'] for r in RUNS]
cols=[GREY]+[COL[r] for r in RUNS]; lab=['Ground truth']+[LAB[r].replace(' (','\n(') for r in RUNS]
axs[0].bar(range(5),vals,color=cols); [axs[0].text(i,v+1,f'{v:.1f}%',ha='center',fontsize=8) for i,v in enumerate(vals)]
axs[0].set_xticks(range(5),lab,fontsize=7); axs[0].set_ylabel('% of queries'); axs[0].set_title('Top-1 window ends at the video end (≤2 s)',fontsize=10)
L=['mAP_short','mAP_middle','mAP_long']; x=np.arange(3)
for i,r in enumerate(RUNS): axs[1].bar(x+(i-1.5)*0.2,[S['runs'][r][k] for k in L],0.18,color=COL[r],hatch=None if 'e50' in r else '//',edgecolor='white',label=LAB[r])
axs[1].set_xticks(x,['short (<10 s)','middle (10–30 s)','long (>30 s)']); axs[1].set_ylabel('mAP avg (%)'); axs[1].legend(frameon=False,fontsize=7); axs[1].set_title('mAP by length of the true moment',fontsize=10)
save('R8_bias_and_moment_length.png','Source: our runs, official evaluator (length buckets), QVHighlights val')
# R7 timelines
P={}
for r in ['hmnet_baseline_fix','hmnet_setg_fix']:
    P[r]={json.loads(l)['qid']:json.loads(l)['pred_relevant_windows'][0] for l in open(R+r+'/predictions_official.jsonl')}
G={json.loads(l)['qid']:json.loads(l) for l in open('/home/claude/ext/moment_detr/data/highlight_val_release.jsonl')}
Q=[(2043,'fixed'),(5266,'fixed'),(4758,'random'),(2570,'random'),(5587,'random'),(4242,'hand-picked')]
def iou(a,b): i=max(0,min(a[1],b[1])-max(a[0],b[0])); u=max(a[1],b[1])-min(a[0],b[0]); return i/u if u>0 else 0
fig,axs=plt.subplots(len(Q),1,figsize=(10,1.55*len(Q)))
for ax,(q,kind) in zip(axs,Q):
    g=G[q]; d=g['duration']; ax.set_xlim(0,d); ax.set_ylim(-0.5,2.6); ax.axis('off')
    rows=[('Ground truth',[w for w in g['relevant_windows']],'#b9b4e0',None),('HM-Net',[P['hmnet_baseline_fix'][q][:2]],C[2],P['hmnet_baseline_fix'][q][2]),('HM-Net + SE-TG',[P['hmnet_setg_fix'][q][:2]],C[6],P['hmnet_setg_fix'][q][2])]
    for k,(n,ws,c,sc) in enumerate(rows):
        y=2-k; ax.add_patch(Rectangle((0,y-0.32),d,0.64,fc='#f6f6f6',ec='#dddddd'))
        for a,b in ws: ax.add_patch(Rectangle((a,y-0.3),b-a,0.6,fc=c,ec='none'))
        txt=n if sc is None else f"{n}  [{ws[0][0]:.0f}–{ws[0][1]:.0f}s]  IoU {max(iou(ws[0],w) for w in g['relevant_windows']):.2f}"
        ax.text(-2,y,txt,ha='right',va='center',fontsize=7.5)
    ax.set_title(f'qid {q} ({kind}): "{g["query"][:95]}"',fontsize=8,loc='left')
fig.suptitle('Top-1 predictions vs ground truth (corrected settings, best checkpoints); 0–150 s timeline',fontsize=10)
plt.subplots_adjust(left=0.3); save('R7_prediction_timelines.png','Source: our runs; fixed queries 2043/5266, random (seed 7) 4758/2570/5587, hand-picked 4242 (not representative)')
print(sorted(os.listdir(O)))
