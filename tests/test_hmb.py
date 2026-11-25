import torch
from src.models.hmb_block import HMBBlock

def test_hmb_shapes_masks_and_backward():
 m=HMBBlock(dim=16,heads=4,state_dim=4,max_seq_len=20,dropout=0,drop_path=0)
 x=torch.randn(2,7,16,requires_grad=True); c=torch.randn(2,5,16)
 mm=torch.tensor([[1]*7,[0]*7],dtype=torch.bool); cm=torch.tensor([[1]*5,[0]*5],dtype=torch.bool)
 y=m(x,c,mm,cm); assert y.shape==x.shape and torch.isfinite(y).all(); y.sum().backward(); assert torch.isfinite(x.grad).all()
 assert y[1].abs().max()==0

def test_position_bias_inactive_at_or_above_threshold():
 m=HMBBlock(dim=16,heads=4,state_dim=4,max_seq_len=5,dropout=0,drop_path=0)
 x=torch.randn(1,6,16); c=torch.randn(1,3,16)
 with torch.no_grad():
  y1=m(x,c)
  m.gamma_bias.fill_(0.0)
  y2=m(x,c)
 # At length >= threshold the positional table is not consulted.
 assert torch.allclose(y1,y2,atol=1e-6)
