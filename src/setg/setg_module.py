from torch import nn
from .subevent_embedding import pool_subevent_embeddings
from .alignment import align_subevents_to_video
from .gated_integration import GatedIntegration

class SETGModule(nn.Module):
    def __init__(self,hidden_dim=256,max_subevents=4,gate_init=0.0):
        super().__init__(); self.max_subevents=max_subevents; self.integration=GatedIntegration(hidden_dim,max_subevents,gate_init)
    def forward(self,video,text,text_mask,video_mask,token_spans,subevent_mask):
        emb=pool_subevent_embeddings(text,text_mask,token_spans,subevent_mask)
        if emb.shape[1]<self.max_subevents:
            pad=emb.new_zeros((emb.shape[0],self.max_subevents-emb.shape[1],emb.shape[-1])); emb=__import__('torch').cat([emb,pad],1)
            mask_pad=subevent_mask.new_zeros((subevent_mask.shape[0],self.max_subevents-subevent_mask.shape[1])); subevent_mask=__import__('torch').cat([subevent_mask,mask_pad],1)
        elif emb.shape[1]>self.max_subevents:
            emb=emb[:,:self.max_subevents]; subevent_mask=subevent_mask[:,:self.max_subevents]
        attn,centers=align_subevents_to_video(video,emb,video_mask,subevent_mask)
        fused=self.integration(video,attn,subevent_mask)
        return fused,{"subevent_alignments":attn,"subevent_centers":centers,"subevent_mask":subevent_mask}
