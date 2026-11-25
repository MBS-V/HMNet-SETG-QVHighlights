import math
import torch

def align_subevents_to_video(video_features, event_embeddings, video_mask, subevent_mask):
    """Returns attention [B,K,L] and normalized soft centers [B,K].

    Computed in fp32 with autocast disabled: under fp16 AMP the dot products can overflow and the
    1e-8 clamp underflows to 0, which produced NaN outputs (observed on MFS, SE-TG epoch 5).
    """
    with torch.autocast(device_type=video_features.device.type, enabled=False):
        return _align_fp32(video_features.float(), event_embeddings.float(), video_mask, subevent_mask)


def _align_fp32(video_features, event_embeddings, video_mask, subevent_mask):
    d=video_features.shape[-1]
    scores=torch.einsum('bld,bkd->bkl',video_features,event_embeddings)/math.sqrt(d)
    scores=scores.masked_fill(~video_mask[:,None,:],torch.finfo(scores.dtype).min)
    attn=scores.softmax(-1)
    attn=attn*video_mask[:,None,:].to(attn.dtype)
    attn=attn/attn.sum(-1,keepdim=True).clamp_min(1e-8)
    lengths=video_mask.sum(-1).clamp_min(1)
    times=torch.arange(video_features.shape[1],device=video_features.device,dtype=video_features.dtype)[None,:]
    denom=(lengths-1).clamp_min(1).to(video_features.dtype)[:,None]
    normalized=times/denom
    centers=(attn*normalized[:,None,:]).sum(-1)
    attn=attn*subevent_mask[:,:,None].to(attn.dtype)
    centers=centers*subevent_mask.to(centers.dtype)
    return attn,centers
