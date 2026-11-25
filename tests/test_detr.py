import torch
from src.models.detr_head import TemporalDETRHead

def test_detr_outputs():
 m=TemporalDETRHead(dim=16,heads=4,num_queries=5,decoder_layers=1,dropout=0)
 x=torch.randn(2,8,16); mask=torch.tensor([[1]*8,[1]*6+[0]*2],dtype=torch.bool); o=m(x,mask)
 assert o['pred_logits'].shape==(2,5,2) and o['pred_spans'].shape==(2,5,2)
 assert ((o['pred_spans']>=0)&(o['pred_spans']<=1)).all()
