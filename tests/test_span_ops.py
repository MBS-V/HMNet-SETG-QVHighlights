import torch
from src.utils.span_ops import cw_to_se,se_to_cw,pairwise_iou,pairwise_giou,elementwise_iou

def test_roundtrip():
 x=torch.tensor([[.5,.4],[.2,.1]]); assert torch.allclose(se_to_cw(cw_to_se(x)),x)
def test_iou():
 a=torch.tensor([[38.,62.]]); b=torch.tensor([[42.,58.]])
 assert abs(pairwise_iou(a,b).item()-16/24)<1e-6
 assert abs(elementwise_iou(a,b).item()-16/24)<1e-6
def test_giou_disjoint():
 a=torch.tensor([[0.,10.]]); b=torch.tensor([[20.,30.]])
 assert pairwise_giou(a,b).item()<0
