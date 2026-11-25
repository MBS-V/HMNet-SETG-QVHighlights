import torch

def pool_subevent_embeddings(text_features, text_mask, token_spans, subevent_mask):
    """Masked mean per clause. text_features [B,N,D], token_spans [B,K,2]."""
    b,k,_=token_spans.shape; d=text_features.shape[-1]
    pooled=text_features.new_zeros((b,k,d))
    for bi in range(b):
        for ki in range(k):
            if not bool(subevent_mask[bi,ki]): continue
            start,end=map(int,token_spans[bi,ki].tolist())
            start=max(0,min(start,text_features.shape[1])); end=max(start,min(end,text_features.shape[1]))
            m=text_mask[bi,start:end]
            if end>start and m.any(): pooled[bi,ki]=(text_features[bi,start:end]*m[:,None]).sum(0)/m.sum().clamp_min(1)
    return pooled
