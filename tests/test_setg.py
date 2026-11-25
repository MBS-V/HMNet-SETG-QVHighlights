import torch
from src.setg.alignment import align_subevents_to_video
from src.setg.gated_integration import GatedIntegration
from src.losses.subevent_losses import containment_loss,diversity_loss

def test_alignment_distribution_and_masks():
 v=torch.randn(2,5,8); e=torch.randn(2,3,8); vm=torch.tensor([[1]*5,[1,1,1,0,0]],dtype=torch.bool); sm=torch.tensor([[1,1,0],[1,0,0]],dtype=torch.bool)
 a,c=align_subevents_to_video(v,e,vm,sm)
 assert torch.allclose(a[0,:2].sum(-1),torch.ones(2)); assert a[1,:,3:].abs().sum()==0
 assert a[:,2].abs().sum()==0

def test_zero_gate_is_exact_identity():
 g=GatedIntegration(8,3,gate_init=0.0); x=torch.randn(2,5,8); a=torch.softmax(torch.randn(2,3,5),-1); m=torch.ones(2,3,dtype=torch.bool)
 assert g.gamma_se.item()==0 and torch.equal(g(x,a,m),x)

def test_losses_k1_finite_zero_diversity():
 a=torch.tensor([[[.2,.3,.5]]],requires_grad=True); sm=torch.tensor([[1]],dtype=torch.bool); vm=torch.ones(1,3,dtype=torch.bool)
 ld=diversity_loss(a,sm); ls=containment_loss(a,sm,torch.tensor([[0.,1.]]),vm)
 assert ld.item()==0 and torch.isfinite(ls); (ld+ls).backward()


def test_all_padding_video_and_event_masks():
 v=torch.randn(1,4,6); e=torch.randn(1,2,6); vm=torch.zeros(1,4,dtype=torch.bool); sm=torch.zeros(1,2,dtype=torch.bool)
 a,c=align_subevents_to_video(v,e,vm,sm)
 assert torch.isfinite(a).all() and a.abs().sum()==0 and c.abs().sum()==0
