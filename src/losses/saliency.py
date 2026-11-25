"""Max-margin contrastive saliency ranking between matched and mismatched pairs."""
import torch
import torch.nn.functional as F

def saliency_max_margin_loss(video_memory,query_embedding,saliency_scores,video_mask,margin=0.2,video_ids=None):
    """Rank each matched query-video pair above mismatched videos in the batch.

    `saliency_scores` is the QVHighlights sparse clip annotation expanded to [B,L]
    with -1 for unannotated clips. The positive-pair score is a rating-weighted
    mean over annotated clips; if labels are unavailable it uses the max valid
    timestep similarity. Negative-pair score is the max valid timestep similarity.
    """
    q=F.normalize(query_embedding,dim=-1); v=F.normalize(video_memory,dim=-1)
    pair_scores=torch.einsum('id,jld->ijl',q,v)
    b=pair_scores.shape[0]; losses=[]
    for i in range(b):
        valid_pos=video_mask[i]
        if saliency_scores is not None:
            labels=saliency_scores[i]
            annotated=valid_pos & (labels>=0)
        else:
            labels=None; annotated=torch.zeros_like(valid_pos)
        if annotated.any():
            weights=labels[annotated].clamp_min(0)+1.0
            positive=(pair_scores[i,i,annotated]*weights).sum()/weights.sum().clamp_min(1e-8)
        elif valid_pos.any():
            positive=pair_scores[i,i,valid_pos].max()
        else:
            continue
        for j in range(b):
            if i==j: continue
            if video_ids is not None and video_ids[i]==video_ids[j]: continue
            valid_neg=video_mask[j]
            if not valid_neg.any(): continue
            negative=pair_scores[i,j,valid_neg].max()
            losses.append(F.relu(margin-positive+negative))
    return torch.stack(losses).mean() if losses else pair_scores.sum()*0
