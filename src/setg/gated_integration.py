import torch
from torch import nn

class GatedIntegration(nn.Module):
    def __init__(self,hidden_dim=256,max_subevents=4,gate_init=0.0):
        super().__init__(); self.proj=nn.Linear(max_subevents,hidden_dim); self.gamma_se=nn.Parameter(torch.tensor(float(gate_init)))
    def forward(self,video,alignments,subevent_mask):
        # alignments [B,K,L] -> [B,L,K]; suppress padded event channels explicitly.
        a=alignments.transpose(1,2)*subevent_mask[:,None,:].to(alignments.dtype)
        return video+torch.tanh(self.gamma_se)*self.proj(a)
