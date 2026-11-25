import torch
from src.losses.subevent_losses import containment_loss

def test_containment_uses_union_of_gt_windows():
 a=torch.tensor([[[.45,.05,.05,.45]]],requires_grad=True)
 sm=torch.ones(1,1,dtype=torch.bool); vm=torch.ones(1,4,dtype=torch.bool)
 gt=[torch.tensor([[0.,.2],[.8,1.]])]
 loss=containment_loss(a,sm,gt,vm)
 assert loss<1.0 and torch.isfinite(loss); loss.backward()
