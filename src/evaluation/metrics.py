"""Internal temporal metrics supporting multiple GT windows per query."""
import numpy as np
import torch

def _iou_matrix(pred,gt):
    inter=(torch.minimum(pred[:,None,1],gt[None,:,1])-torch.maximum(pred[:,None,0],gt[None,:,0])).clamp_min(0)
    union=(pred[:,None,1]-pred[:,None,0]).clamp_min(0)+(gt[None,:,1]-gt[None,:,0]).clamp_min(0)-inter
    return torch.where(union>0,inter/union.clamp_min(1e-8),torch.zeros_like(union))

def compute_metrics(pred_spans,gt_spans,logits=None,thresholds=(0.3,0.5,0.7)):
    p=torch.as_tensor(pred_spans,dtype=torch.float32)
    if p.ndim==2: p=p[:,None,:]
    if logits is None: scores=torch.ones(p.shape[:2])
    else: scores=torch.as_tensor(logits).softmax(-1)[...,1]
    if isinstance(gt_spans,torch.Tensor):
        if gt_spans.ndim==2: gts=[gt_spans[i:i+1] for i in range(len(gt_spans))]
        elif gt_spans.ndim==3: gts=[gt_spans[i] for i in range(len(gt_spans))]
    else: gts=[torch.as_tensor(x,dtype=torch.float32) for x in gt_spans]
    order=scores.argsort(dim=1,descending=True); top=order[:,0]
    top_ious=[]; all_ious=[]
    for i,gt in enumerate(gts):
        mat=_iou_matrix(p[i],gt)
        all_ious.append(mat); top_ious.append(mat[top[i]].max())
    top_iou=torch.stack(top_ious).cpu().numpy() if top_ious else np.array([])
    out={f'R1@{t:.1f}':float(np.mean(top_iou>=t)*100) if len(top_iou) else 0.0 for t in thresholds}
    out['mIoU']=float(top_iou.mean()*100) if len(top_iou) else 0.0
    def mean_ap(threshold):
        aps=[]
        for i,mat in enumerate(all_ious):
            ranked=mat[order[i]]; matched=set(); tp=[]; fp=[]
            for row in ranked:
                available=[j for j in range(len(gts[i])) if j not in matched]
                if available:
                    best_gt=max(available,key=lambda j:float(row[j])); best=float(row[best_gt])
                else:
                    best_gt=-1; best=-1.0
                if best>=threshold: matched.add(best_gt); tp.append(1); fp.append(0)
                else: tp.append(0); fp.append(1)
            if not tp: aps.append(0.0); continue
            tp_c=np.cumsum(tp); fp_c=np.cumsum(fp); precision=tp_c/np.maximum(tp_c+fp_c,1)
            aps.append(float((precision*np.asarray(tp)).sum()/max(1,len(gts[i]))))
        return float(np.mean(aps)*100) if aps else 0.0
    out['mAP@0.50']=mean_ap(0.50); out['mAP@0.75']=mean_ap(0.75)
    out['mAP_avg']=float(np.mean([mean_ap(t) for t in np.arange(0.5,0.951,0.05)]))
    return out
