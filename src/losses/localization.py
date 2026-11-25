"""Hungarian set matching and DETR-style localization objective."""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from ..utils.span_ops import cw_to_se, se_to_cw, pairwise_giou

def hungarian_match(pred_logits,pred_spans,targets, cost_class=1.0,cost_l1=5.0,cost_giou=2.0):
    matches=[]
    for b,t in enumerate(targets):
        gt=t['spans'].to(pred_spans.device)
        if gt.numel()==0:
            matches.append((torch.empty(0,dtype=torch.long,device=pred_spans.device),torch.empty(0,dtype=torch.long,device=pred_spans.device))); continue
        prob=pred_logits[b].softmax(-1)[:,1]
        cls=-prob[:,None].expand(-1,gt.shape[0])
        l1=torch.cdist(pred_spans[b],se_to_cw(gt),p=1)
        giou=pairwise_giou(cw_to_se(pred_spans[b]),gt)
        cost=cost_class*cls+cost_l1*l1-cost_giou*giou
        row,col=linear_sum_assignment(cost.detach().cpu().numpy())
        matches.append((torch.as_tensor(row,device=pred_spans.device),torch.as_tensor(col,device=pred_spans.device)))
    return matches

def localization_loss(outputs,targets,weight_ce=6.,weight_l1=10.,weight_giou=1.,eos_coef=1.0):
    logits,spans=outputs['pred_logits'],outputs['pred_spans']; matches=hungarian_match(logits,spans,targets)
    labels=torch.zeros(logits.shape[:2],dtype=torch.long,device=logits.device)
    matched_pred=[]; matched_gt=[]
    for b,(pi,gi) in enumerate(matches):
        if pi.numel(): labels[b,pi]=1; matched_pred.append(spans[b,pi]); matched_gt.append(targets[b]['spans'].to(spans.device)[gi])
    ce=F.cross_entropy(logits.transpose(1,2),labels,weight=torch.tensor([eos_coef,1.0],device=logits.device,dtype=logits.dtype) if eos_coef!=1.0 else None)
    if matched_pred:
        p=torch.cat(matched_pred); g=torch.cat(matched_gt)
        l1=F.l1_loss(p,se_to_cw(g))
        giou=1-pairwise_giou(cw_to_se(p),g).diag().mean()
    else:
        l1=spans.sum()*0; giou=spans.sum()*0
    total=weight_ce*ce+weight_l1*l1+weight_giou*giou
    return total,{'loss_ce':ce.detach(),'loss_l1':l1.detach(),'loss_giou':giou.detach()}
