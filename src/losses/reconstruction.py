"""MESM-derived multi-level reconstruction regularizer.

Frame-word MLM shares the first enhancement HMB cross-attention weights. The
segment-sentence branch uses GT-pooled video segments and a batch contrastive
objective with same-video, high-IoU positives, following MESM's formulation.
"""
import torch
import math
from torch import nn
from torch.nn import functional as F
from ..utils.span_ops import pairwise_iou

class MESMReconstruction(nn.Module):
    def __init__(self, dim=256, vocab_size=49408, mask_fraction=1/3, temperature=0.07, positive_iou=0.9, heads=8):
        super().__init__()
        self.vocab_size=int(vocab_size); self.mask_fraction=float(mask_fraction)
        self.temperature=float(temperature); self.positive_iou=float(positive_iou)
        self.mask_token=nn.Parameter(torch.zeros(1,1,dim)); nn.init.normal_(self.mask_token,std=0.02)
        self.word_classifier=nn.Linear(dim,self.vocab_size)
        self.sentence_mask=nn.Parameter(torch.zeros(1,1,dim)); nn.init.normal_(self.sentence_mask,std=0.02)
        self.segment_sentence_attn=nn.MultiheadAttention(dim,heads,batch_first=True)

    def frame_word_loss(self, video, text, video_mask, text_mask, token_ids, shared_cross_attention, video_norm, text_norm, token_loss_mask=None):
        zero=text.sum()*0
        if token_ids is None: return zero
        valid=(token_loss_mask if token_loss_mask is not None else text_mask) & token_ids.ge(0) & token_ids.lt(self.vocab_size)
        random=torch.rand(valid.shape,device=valid.device)
        masked=valid & (random<self.mask_fraction)
        # Ensure at least one masked valid word for each non-empty query.
        for b in range(valid.shape[0]):
            ids=torch.nonzero(valid[b],as_tuple=False).flatten()
            if ids.numel() and not masked[b].any(): masked[b,ids[torch.randint(ids.numel(),(1,),device=ids.device)]]=True
        if not masked.any(): return zero
        q=text_norm(text).clone()
        q=torch.where(masked.unsqueeze(-1),self.mask_token.to(q.dtype),q)
        memory=video_norm(video)
        empty=~video_mask.any(1); safe_mask=video_mask.clone()
        if empty.any(): safe_mask[empty,0]=True; memory=memory.clone(); memory[empty,0]=0
        reconstructed,_=shared_cross_attention(q,memory,memory,key_padding_mask=~safe_mask,need_weights=False)
        logits=self.word_classifier(reconstructed)
        return F.cross_entropy(logits[masked],token_ids[masked].long())

    def _segment_pool(self, video, video_mask, gt_spans):
        pooled=[]
        for b in range(video.shape[0]):
            valid=torch.nonzero(video_mask[b],as_tuple=False).flatten()
            if not valid.numel(): pooled.append(video[b].sum(0)*0); continue
            length=int(valid.numel())
            windows=gt_spans[b] if isinstance(gt_spans,(list,tuple)) else gt_spans[b].view(1,2)
            segments=[]
            for s,e in windows.tolist():
                lo=max(0,min(length-1,math.floor(s*(length-1))))
                hi=max(lo+1,min(length,math.ceil(e*(length-1))+1))
                segments.append(video[b,valid[lo:hi]].mean(0))
            pooled.append(torch.stack(segments).mean(0))
        return torch.stack(pooled)

    def segment_sentence_loss(self, video, text, video_mask, text_mask, gt_spans, video_ids=None):
        zero=text.sum()*0
        if gt_spans is None: return zero
        weights=text_mask.unsqueeze(-1).to(text.dtype)
        mask_query=self.sentence_mask.to(text.dtype).expand(text.shape[0],-1,-1)
        memory=video
        safe_mask=video_mask.clone(); empty=~safe_mask.any(1)
        if empty.any(): safe_mask[empty,0]=True; memory=memory.clone(); memory[empty,0]=0
        generated,_=self.segment_sentence_attn(mask_query,memory,memory,key_padding_mask=~safe_mask,need_weights=False)
        # MESM concatenates the generated complementary token with original query tokens.
        enhanced=torch.cat([generated,text*weights],dim=1)
        # MESM (model/criterion.py::loss_rec_ss): masked MEAN over [generated; words], L2-normalise both
        # sides, cosine similarity / tau (MESM tau=0.5). Computed in fp32 (safe under AMP).
        enh_w=torch.cat([weights.new_ones((weights.shape[0],1,1)),weights],dim=1)
        sentence=(enhanced*enh_w).sum(1)/enh_w.sum(1).clamp_min(1.0)
        segment=self._segment_pool(video,video_mask,gt_spans)
        scores=(F.normalize(sentence.float(),dim=-1)@F.normalize(segment.float(),dim=-1).t())/max(self.temperature,1e-6)
        b=len(gt_spans); positive=torch.eye(b,dtype=torch.bool,device=video.device)
        if video_ids is not None:
            same=torch.tensor([[video_ids[i]==video_ids[j] for j in range(b)] for i in range(b)],device=video.device,dtype=torch.bool)
            windows=[gt_spans[i] if isinstance(gt_spans,(list,tuple)) else gt_spans[i].view(1,2) for i in range(b)]
            overlaps=torch.eye(b,dtype=torch.bool,device=video.device)
            for i in range(b):
                for j in range(b):
                    if i!=j: overlaps[i,j]=pairwise_iou(windows[i],windows[j]).max()>self.positive_iou
            positive |= same & overlaps
        log_den=torch.logsumexp(scores,dim=1)
        log_num=torch.logsumexp(scores.masked_fill(~positive,float('-inf')),dim=1)
        return (log_den-log_num).mean()

    def forward(self, video, text, video_mask, text_mask, gt_spans=None, token_ids=None, video_ids=None, shared_cross_attention=None, video_norm=None, text_norm=None, token_loss_mask=None):
        if shared_cross_attention is None or video_norm is None or text_norm is None:
            raise ValueError('MESM reconstruction requires the shared first-HMB cross-attention and its modality norms')
        l_fw=self.frame_word_loss(video,text,video_mask,text_mask,token_ids,shared_cross_attention,video_norm,text_norm,token_loss_mask)
        l_ss=self.segment_sentence_loss(video,text,video_mask,text_mask,gt_spans,video_ids)
        return {'loss_fw':l_fw,'loss_ss':l_ss,'loss_reconstruction':l_fw+l_ss}
