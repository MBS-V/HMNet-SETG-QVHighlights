import torch
from src.losses.saliency import saliency_max_margin_loss

def test_saliency_margin_ranks_positive_pair_over_negative_video():
 v=torch.tensor([[[0.,1.],[0.,1.]],[[1.,0.],[1.,0.]]],requires_grad=True)
 q=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True)
 labels=torch.tensor([[4.,4.],[4.,4.]])
 mask=torch.ones(2,2,dtype=torch.bool)
 loss=saliency_max_margin_loss(v,q,labels,mask,margin=.2,video_ids=['v1','v2'])
 assert torch.isfinite(loss) and loss>0
 loss.backward(); assert torch.isfinite(v.grad).all()

def test_saliency_without_negative_pairs_is_differentiable_zero():
 v=torch.randn(1,3,4,requires_grad=True); q=torch.randn(1,4,requires_grad=True)
 loss=saliency_max_margin_loss(v,q,torch.full((1,3),-1.),torch.ones(1,3,dtype=torch.bool),video_ids=['v'])
 assert loss.item()==0; loss.backward()
