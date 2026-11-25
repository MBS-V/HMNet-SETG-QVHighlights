"""Two-stage hierarchical fusion encoders."""
from torch import nn
from .hmb_block import HMBBlock

class FrameWordEnhancementEncoder(nn.Module):
    def __init__(self, dim=256, layers=2, **kwargs):
        super().__init__(); self.blocks=nn.ModuleList([HMBBlock(dim=dim, **kwargs) for _ in range(layers)])
    def forward(self, video, text, video_mask, text_mask):
        for block in self.blocks: video=block(video,text,video_mask,text_mask)
        return video

class SymmetricCoFusionEncoder(nn.Module):
    def __init__(self, dim=256, layers=2, **kwargs):
        super().__init__()
        self.video_blocks=nn.ModuleList([HMBBlock(dim=dim, **kwargs) for _ in range(layers)])
        self.text_blocks=nn.ModuleList([HMBBlock(dim=dim, **kwargs) for _ in range(layers)])
    def forward(self, video, text, video_mask, text_mask):
        for vb,tb in zip(self.video_blocks,self.text_blocks):
            old_v,old_t=video,text
            # Simultaneous updates: each direction consumes the previous layer states.
            new_v=vb(old_v,old_t,video_mask,text_mask)
            new_t=tb(old_t,old_v,text_mask,video_mask)
            video,text=new_v,new_t
        weights=text_mask.unsqueeze(-1).to(text.dtype)
        q=(text*weights).sum(1)/weights.sum(1).clamp_min(1.0)
        return video+q.unsqueeze(1), text
