"""HM-Net baseline and optional SE-TG extension."""
import torch
from torch import nn
from .encoders import FrameWordEnhancementEncoder, SymmetricCoFusionEncoder
from .detr_head import TemporalDETRHead
from ..setg.setg_module import SETGModule
from ..losses.reconstruction import MESMReconstruction

class HMNet(nn.Module):
    def __init__(self, video_dim=2816, text_dim=512, hidden_dim=256, heads=8, state_dim=32,
                 enhancement_layers=2, cofusion_layers=2, num_queries=10, decoder_layers=2,
                 max_seq_len=200, dropout=0.1, drop_path=0.0, use_setg=False, max_subevents=4, use_reconstruction=True, vocab_size=49408, reconstruction_mask_fraction=1/3, reconstruction_temperature=0.07, reconstruction_positive_iou=0.9):
        super().__init__()
        self.video_proj=nn.Linear(video_dim,hidden_dim); self.text_proj=nn.Linear(text_dim,hidden_dim)
        block_kwargs=dict(heads=heads,state_dim=state_dim,max_seq_len=max_seq_len,dropout=dropout,drop_path=drop_path)
        self.enhancement=FrameWordEnhancementEncoder(hidden_dim,enhancement_layers,**block_kwargs)
        self.cofusion=SymmetricCoFusionEncoder(hidden_dim,cofusion_layers,**block_kwargs)
        self.setg=SETGModule(hidden_dim,max_subevents) if use_setg else None
        self.reconstruction=MESMReconstruction(hidden_dim,vocab_size=vocab_size,heads=heads,mask_fraction=reconstruction_mask_fraction,temperature=reconstruction_temperature,positive_iou=reconstruction_positive_iou) if use_reconstruction else None
        self.head=TemporalDETRHead(hidden_dim,heads,num_queries,decoder_layers,dropout)

    def forward(self, video, text, video_mask, text_mask, subevent_spans=None, subevent_mask=None, token_ids=None, gt_spans=None, video_ids=None, token_loss_mask=None):
        v=self.video_proj(video); q=self.text_proj(text)
        reconstruction_aux={}
        if self.reconstruction is not None and self.training and gt_spans is not None:
            reconstruction_aux=self.reconstruction(v,q,video_mask,text_mask,gt_spans,token_ids,video_ids,
                shared_cross_attention=self.enhancement.blocks[0].cross_attn,
                video_norm=self.enhancement.blocks[0].main_norm, text_norm=self.enhancement.blocks[0].ctx_norm, token_loss_mask=token_loss_mask)
        v=self.enhancement(v,q,video_mask,text_mask)
        v,q=self.cofusion(v,q,video_mask,text_mask)
        aux={}
        if self.setg is not None:
            if subevent_spans is None:
                # K=1 fallback covers the full query; parser spans can be supplied by collate.
                subevent_spans=torch.tensor([[[0,text.shape[1]]]],device=text.device).expand(text.shape[0],-1,-1)
                subevent_mask=torch.ones((text.shape[0],1),dtype=torch.bool,device=text.device)
            v,aux=self.setg(v,q,text_mask,video_mask,subevent_spans,subevent_mask)
        out=self.head(v,video_mask); out.update(aux); out.update(reconstruction_aux)
        q_weights=text_mask.unsqueeze(-1).to(q.dtype)
        out['video_memory']=v
        out['query_embedding']=(q*q_weights).sum(1)/q_weights.sum(1).clamp_min(1.0)
        return out
