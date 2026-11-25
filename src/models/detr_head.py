"""DETR-style temporal set prediction head (decoder over fused video memory)."""
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, din, dh, dout, layers=3):
        super().__init__(); dims=[din]+[dh]*(layers-1)+[dout]
        self.layers=nn.ModuleList([nn.Linear(a,b) for a,b in zip(dims[:-1],dims[1:])])
    def forward(self,x):
        for i,layer in enumerate(self.layers):
            x=layer(x)
            if i<len(self.layers)-1: x=torch.relu(x)
        return x

class TemporalDETRHead(nn.Module):
    def __init__(self, dim=256, heads=8, num_queries=10, decoder_layers=2, dropout=0.1):
        super().__init__()
        layer=nn.TransformerDecoderLayer(dim,heads,dim_feedforward=dim*4,dropout=dropout,batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(layer,decoder_layers,norm=nn.LayerNorm(dim))
        self.query_embed=nn.Embedding(num_queries,dim)
        self.class_head=nn.Linear(dim,2); self.span_head=MLP(dim,dim,2,3)
    def forward(self,memory,video_mask):
        q=self.query_embed.weight.unsqueeze(0).expand(memory.shape[0],-1,-1)
        h=self.decoder(q,memory,memory_key_padding_mask=~video_mask)
        return {"pred_logits":self.class_head(h),"pred_spans":self.span_head(h).sigmoid(),"query_features":h}
