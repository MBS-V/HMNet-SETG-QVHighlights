"""Synthetic-only integration smoke test. Never substitutes for real-data evaluation."""
import sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.hmnet import HMNet
from src.losses.total_loss import compute_total_loss

torch.manual_seed(7); device='cuda' if torch.cuda.is_available() else 'cpu'
base=HMNet(video_dim=32,text_dim=16,hidden_dim=32,heads=4,state_dim=4,enhancement_layers=1,cofusion_layers=1,num_queries=5,decoder_layers=1,dropout=0).to(device)
extended=HMNet(video_dim=32,text_dim=16,hidden_dim=32,heads=4,state_dim=4,enhancement_layers=1,cofusion_layers=1,num_queries=5,decoder_layers=1,dropout=0,use_setg=True,max_subevents=3).to(device)
extended.load_state_dict(base.state_dict(),strict=False); base.eval(); extended.eval()
v=torch.randn(2,9,32,device=device); q=torch.randn(2,6,16,device=device)
vm=torch.ones(2,9,dtype=torch.bool,device=device); vm[1,7:]=False
tm=torch.ones(2,6,dtype=torch.bool,device=device); spans=torch.tensor([[[0,3],[3,6]],[[0,6],[0,0]]],device=device); sm=torch.tensor([[1,1],[1,0]],dtype=torch.bool,device=device)
with torch.no_grad():
 a=base(v,q,vm,tm); b=extended(v,q,vm,tm,spans,sm)
assert torch.allclose(a['pred_logits'],b['pred_logits'],atol=1e-6)
assert torch.allclose(a['pred_spans'],b['pred_spans'],atol=1e-6)
extended.train(); out=extended(v,q,vm,tm,spans,sm)
targets=[{'spans':torch.tensor([[.2,.7]])},{'spans':torch.tensor([[.1,.6]])}]
loss,parts=compute_total_loss(out,targets,torch.tensor([[.2,.7],[.1,.6]],device=device),vm)
loss.backward()
assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in extended.parameters())
print('synthetic integration smoke test passed; device=',device,'loss=',float(loss.detach()),'Mamba CUDA kernel=',__import__('src.models.mamba_block',fromlist=['MAMBA_SSM_AVAILABLE']).MAMBA_SSM_AVAILABLE)
