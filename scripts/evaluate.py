"""Evaluate one checkpoint: official-format predictions (seconds) + internal metrics.

Writes to --output-dir:
  predictions_official.jsonl   {"qid","query","vid","pred_relevant_windows":[[start_s,end_s,score],...]}
  gt_official.jsonl            ground truth for exactly the evaluated rows (official format)
  metrics_internal.json/.md    internal R1@{0.3,0.5,0.7}, mIoU, mAP (NOT the benchmark numbers)
Report benchmark numbers ONLY from the official Moment-DETR evaluator:
  python moment_detr/standalone_eval/eval.py --submission_path <dir>/predictions_official.jsonl \
      --gt_path <dir>/gt_official.jsonl --save_path <dir>/metrics_official.json
"""
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.data.dataset import FeatureMomentDataset
from src.data.collate import collate_moments
from src.models.hmnet import HMNet
from src.engine.trainer import predict_windows
from src.evaluation.metrics import compute_metrics

def _qid(q):
    return int(q) if str(q).isdigit() else q

p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--manifest',required=True); p.add_argument('--feature-root')
p.add_argument('--checkpoint',required=True); p.add_argument('--output-dir',default='outputs/evaluation'); p.add_argument('--batch-size',type=int,default=32); a=p.parse_args()
cfg=load_config(a.config); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds=FeatureMomentDataset(a.manifest,a.feature_root or cfg.data.feature_dir,cfg.data.max_subevents,l2_normalize=bool(cfg.data.get('l2_normalize',False)))
dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=2,collate_fn=lambda x:collate_moments(x,cfg.data.max_subevents))
m=HMNet(video_dim=cfg.data.video_dim,text_dim=cfg.data.text_dim,hidden_dim=cfg.model.hidden_dim,heads=cfg.model.heads,state_dim=cfg.model.state_dim,
 enhancement_layers=cfg.model.enhancement_layers,cofusion_layers=cfg.model.cofusion_layers,num_queries=cfg.model.num_queries,decoder_layers=cfg.model.decoder_layers,
 max_seq_len=cfg.model.max_seq_len,dropout=cfg.model.dropout,drop_path=cfg.model.drop_path,use_setg=cfg.model.use_setg,max_subevents=cfg.data.max_subevents,use_reconstruction=cfg.model.use_reconstruction,vocab_size=cfg.data.vocab_size,reconstruction_mask_fraction=cfg.reconstruction.mask_fraction,reconstruction_temperature=cfg.reconstruction.temperature,reconstruction_positive_iou=cfg.reconstruction.positive_iou).to(dev)
s=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
m.load_state_dict(s.get('model',s))   # strict: fails if --config does not match the run (e.g. use_setg)
print(f"loaded {a.checkpoint} (epoch {s.get('epoch')}), use_setg={cfg.model.use_setg}")
preds=predict_windows(m,dl,dev)
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
rows_by_qid={str(r.get('qid')):r for r in ds.rows}
with open(out/'predictions_official.jsonl','w') as f, open(out/'gt_official.jsonl','w') as g:
    for p_ in preds:
        d=p_['duration']; r=rows_by_qid[p_['qid']]
        wins=[[round(float(st)*d,2),round(float(en)*d,2),round(float(sc),4)] for (st,en),sc in zip(p_['spans'].tolist(),p_['scores'].tolist())]
        f.write(json.dumps({'qid':_qid(p_['qid']),'query':p_['query'],'vid':p_['video_id'],'pred_relevant_windows':wins})+'\n')
        g.write(json.dumps({'qid':_qid(p_['qid']),'query':r.get('query'),'duration':r.get('duration'),'vid':p_['video_id'],
                            'relevant_windows':r.get('relevant_windows',r.get('timestamps'))})+'\n')
logits=[torch.stack([1-p_['scores'],p_['scores']],-1).clamp_min(1e-8).log() for p_ in preds]
metrics=compute_metrics(torch.stack([p_['spans'] for p_ in preds]),[p_['gt'] for p_ in preds],torch.stack(logits))
(out/'metrics_internal.json').write_text(json.dumps({'metrics':metrics,'n_queries':len(preds),'use_setg':bool(cfg.model.use_setg),
    'note':'internal metrics for sanity/model selection; report the official evaluator output'},indent=2))
(out/'metrics_internal.md').write_text('# Internal metrics (not benchmark numbers)\n\n'+'\n'.join(f'- {k}: {v:.2f}' for k,v in metrics.items())+'\n')
print(json.dumps(metrics,indent=1)); print(f'wrote {out}/predictions_official.jsonl and gt_official.jsonl ({len(preds)} queries)')
