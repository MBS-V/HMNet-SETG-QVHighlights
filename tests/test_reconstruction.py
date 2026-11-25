import torch
from torch import nn
from src.losses.reconstruction import MESMReconstruction

def test_mesm_frame_word_and_segment_sentence_losses():
 torch.manual_seed(1); b,n,l,d=2,5,7,16
 module=MESMReconstruction(dim=d,vocab_size=50,mask_fraction=1.0,heads=4)
 shared=nn.MultiheadAttention(d,4,batch_first=True); vn=nn.LayerNorm(d); tn=nn.LayerNorm(d)
 v=torch.randn(b,l,d,requires_grad=True); q=torch.randn(b,n,d,requires_grad=True)
 vm=torch.ones(b,l,dtype=torch.bool); tm=torch.ones(b,n,dtype=torch.bool)
 ids=torch.randint(0,50,(b,n)); gt=torch.tensor([[.1,.6],[.2,.8]])
 out=module(v,q,vm,tm,gt,ids,['v1','v2'],shared,vn,tn)
 assert torch.isfinite(out['loss_reconstruction']) and out['loss_fw']>0
 out['loss_reconstruction'].backward(); assert torch.isfinite(v.grad).all() and torch.isfinite(q.grad).all()

def test_reconstruction_handles_missing_token_ids():
 module=MESMReconstruction(dim=8,vocab_size=20,heads=2)
 shared=nn.MultiheadAttention(8,2,batch_first=True); norm=nn.LayerNorm(8)
 v=torch.randn(2,4,8); q=torch.randn(2,3,8); vm=torch.ones(2,4,dtype=torch.bool); tm=torch.ones(2,3,dtype=torch.bool)
 out=module(v,q,vm,tm,torch.tensor([[0.,1.],[0.,1.]]),None,['a','b'],shared,norm,norm)
 assert out['loss_fw'].item()==0 and torch.isfinite(out['loss_ss'])
