import torch

def containment_loss(alignments,subevent_mask,gt_spans,video_mask):
    """GT spans are normalized [start,end]; alignment channels sum to one."""
    b,k,l=alignments.shape; terms=[]
    lengths=video_mask.sum(-1).clamp_min(1)
    for bi in range(b):
        valid_idx=torch.nonzero(video_mask[bi],as_tuple=False).flatten()
        if valid_idx.numel()==0: continue
        windows=gt_spans[bi] if isinstance(gt_spans,(list,tuple)) else gt_spans[bi].view(1,2)
        # Clip centers are supervised by the union of all annotated relevant windows.
        times=valid_idx.to(alignments.dtype)/(max(int(lengths[bi])-1,1))
        inside=torch.zeros_like(times,dtype=torch.bool)
        for start,end in windows.tolist(): inside |= (times>=start)&(times<=end)
        if not inside.any():
            # Include the nearest valid frame to any annotated interval for very narrow spans.
            centers=torch.tensor([(a+b)*0.5 for a,b in windows.tolist()],device=times.device,dtype=times.dtype)
            nearest=(times[:,None]-centers[None,:]).abs().min(-1).values
            inside[torch.argmin(nearest)]=True
        for ki in range(k):
            if bool(subevent_mask[bi,ki]):
                mass=alignments[bi,ki,valid_idx[inside]].sum().clamp(1e-8,1.0)
                terms.append(-torch.log(mass))
    return torch.stack(terms).mean() if terms else alignments.sum()*0

def diversity_loss(alignments,subevent_mask):
    terms=[]
    for bi in range(alignments.shape[0]):
        ids=torch.nonzero(subevent_mask[bi],as_tuple=False).flatten()
        k=ids.numel()
        if k<2: continue
        a=alignments[bi,ids]
        overlap=a@a.transpose(0,1)
        terms.append((overlap.sum()-overlap.diagonal().sum())/(k*(k-1)))
    return torch.stack(terms).mean() if terms else alignments.sum()*0
