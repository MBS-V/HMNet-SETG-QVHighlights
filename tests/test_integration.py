import torch
from src.models.hmnet import HMNet
from src.losses.total_loss import compute_total_loss

def test_baseline_setg_initial_equivalence_and_backward():
 torch.manual_seed(4)
 kwargs=dict(video_dim=16,text_dim=12,hidden_dim=16,heads=4,state_dim=4,enhancement_layers=1,cofusion_layers=1,num_queries=3,decoder_layers=1,dropout=0)
 base=HMNet(**kwargs); ext=HMNet(**kwargs,use_setg=True,max_subevents=3); ext.load_state_dict(base.state_dict(),strict=False)
 base.eval(); ext.eval(); v=torch.randn(2,6,16); q=torch.randn(2,4,12); vm=torch.ones(2,6,dtype=torch.bool); tm=torch.ones(2,4,dtype=torch.bool)
 spans=torch.tensor([[[0,2],[2,4]],[[0,4],[0,0]]]); sm=torch.tensor([[1,1],[1,0]],dtype=torch.bool)
 a=base(v,q,vm,tm); b=ext(v,q,vm,tm,spans,sm)
 assert torch.allclose(a['pred_spans'],b['pred_spans'],atol=1e-6)
 ext.train(); out=ext(v,q,vm,tm,spans,sm); loss,_=compute_total_loss(out,[{'spans':torch.tensor([[.2,.8]])},{'spans':torch.tensor([[.1,.6]])}],torch.tensor([[.2,.8],[.1,.6]]),vm)
 loss.backward(); assert torch.isfinite(loss)
