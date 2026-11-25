"""Minimal resumable training engine for manifest-backed data."""
from pathlib import Path
import torch
from ..losses.total_loss import compute_total_loss
from ..utils.checkpoint import CheckpointManager

def train_one_epoch(model,loader,optimizer,device,epoch,scaler=None,grad_accum=1,grad_clip=1.0,amp=False,loss_weights=None):
    model.train(); total=0.; steps=0; skipped=0; optimizer.zero_grad(set_to_none=True)
    for step,b in enumerate(loader):
        v=b['video'].to(device); q=b['text'].to(device); vm=b['video_mask'].to(device); tm=b['text_mask'].to(device)
        spans=b['token_spans'].to(device); sm=b['subevent_mask'].to(device); gt=[x.to(device) for x in b['gt_spans']]
        use_amp=amp and device.type=='cuda'
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
            out=model(v,q,vm,tm,spans,sm,token_ids=b['token_ids'].to(device) if b['token_ids'] is not None else None,gt_spans=gt,video_ids=b['video_ids'],token_loss_mask=b['mlm_mask'].to(device))
            if not (torch.isfinite(out['pred_logits']).all() and torch.isfinite(out['pred_spans']).all()):
                skipped+=1; print(f'[warn] epoch {epoch} step {step}: non-finite model output, batch skipped',flush=True); continue
            loss,parts=compute_total_loss(out,b['targets'],gt,vm,saliency_scores=b['saliency'].to(device) if b['saliency'] is not None else None,video_ids=b['video_ids'],**(loss_weights or {})); loss=loss/grad_accum
        if not torch.isfinite(loss):
            skipped+=1; print(f'[warn] epoch {epoch} step {step}: non-finite loss, batch skipped',flush=True); optimizer.zero_grad(set_to_none=True); continue
        if scaler is not None and scaler.is_enabled(): scaler.scale(loss).backward()
        else: loss.backward()
        if (step+1)%grad_accum==0 or step+1==len(loader):
            remainder=(step+1)%grad_accum
            if remainder:
                actual=remainder
                correction=grad_accum/actual
                for parameter in model.parameters():
                    if parameter.grad is not None: parameter.grad.mul_(correction)
            if scaler is not None and scaler.is_enabled(): scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
            if scaler is not None and scaler.is_enabled(): scaler.step(optimizer); scaler.update()
            else: optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total+=float(loss.detach())*grad_accum; steps+=1
    return {'loss':total/max(1,steps),'steps':steps,'skipped_batches':skipped}

@torch.no_grad()
def evaluate_loss(model,loader,device,loss_weights=None):
    model.eval(); total=0.; n=0
    for b in loader:
        v=b['video'].to(device); q=b['text'].to(device); vm=b['video_mask'].to(device); tm=b['text_mask'].to(device)
        out=model(v,q,vm,tm,b['token_spans'].to(device),b['subevent_mask'].to(device),token_ids=b['token_ids'].to(device) if b['token_ids'] is not None else None,gt_spans=[x.to(device) for x in b['gt_spans']],video_ids=b['video_ids'],token_loss_mask=b['mlm_mask'].to(device))
        loss,_=compute_total_loss(out,b['targets'],[x.to(device) for x in b['gt_spans']],vm,saliency_scores=b['saliency'].to(device) if b['saliency'] is not None else None,video_ids=b['video_ids'],**(loss_weights or {})); total+=float(loss); n+=1
    return {'loss':total/max(1,n),'batches':n}

@torch.no_grad()
def predict_windows(model,loader,device,top_k=10):
    """Run the model and return per-query predictions: normalized [start,end] spans, fg scores, GT."""
    from ..utils.span_ops import cw_to_se
    model.eval(); preds=[]
    for b in loader:
        v=b['video'].to(device); q=b['text'].to(device); vm=b['video_mask'].to(device); tm=b['text_mask'].to(device)
        out=model(v,q,vm,tm,b['token_spans'].to(device),b['subevent_mask'].to(device))
        spans=cw_to_se(out['pred_spans'].float()).clamp(0,1).cpu(); prob=out['pred_logits'].float().softmax(-1)[...,1].cpu()
        for i in range(len(b['qids'])):
            order=prob[i].argsort(descending=True)[:top_k]
            preds.append({'qid':b['qids'][i],'video_id':b['video_ids'][i],'query':b['queries'][i],'duration':float(b['durations'][i]),
                          'spans':spans[i][order],'scores':prob[i][order],'gt':b['gt_spans'][i]})
    return preds

def evaluate_metrics(model,loader,device):
    """Internal R1/mIoU/mAP on the validation loader (model selection only; report the official evaluator)."""
    from ..evaluation.metrics import compute_metrics
    preds=predict_windows(model,loader,device)
    logits=[torch.stack([1-p['scores'],p['scores']],-1).clamp_min(1e-8).log() for p in preds]  # preserves the score order
    return compute_metrics(torch.stack([p['spans'] for p in preds]),[p['gt'] for p in preds],torch.stack(logits))
