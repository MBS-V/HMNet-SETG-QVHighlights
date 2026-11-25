import torch
from src.models.mamba_block import BidirectionalMamba

def test_bimamba_forward_backward():
 m=BidirectionalMamba(16,state_dim=4,prefer_cuda=False); x=torch.randn(2,7,16,requires_grad=True)
 mask=torch.tensor([[1]*7,[1]*5+[0]*2],dtype=torch.bool); y=m(x,mask)
 assert y.shape==x.shape and torch.isfinite(y).all(); y.sum().backward(); assert torch.isfinite(x.grad).all()
def test_independent_directions():
 m=BidirectionalMamba(8,4,prefer_cuda=False)
 assert m.forward_mamba is not m.backward_mamba

def test_multiple_sequence_lengths():
 m=BidirectionalMamba(8,state_dim=32,prefer_cuda=False)
 for length in (1,3,11):
  x=torch.randn(1,length,8); y=m(x)
  assert y.shape==(1,length,8) and torch.isfinite(y).all()
