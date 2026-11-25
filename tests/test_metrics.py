import torch
from src.evaluation.metrics import compute_metrics

def test_metrics_support_multiple_ground_truth_windows():
 pred=torch.tensor([[[.0,.2],[.5,.8],[.3,.4]]])
 gt=[torch.tensor([[0.,.2],[.5,.8]])]
 logits=torch.tensor([[[0.,5.],[0.,4.],[0.,1.]]])
 m=compute_metrics(pred,gt,logits)
 assert m['R1@0.5']==100 and m['mIoU']>99 and m['mAP@0.50']>99
