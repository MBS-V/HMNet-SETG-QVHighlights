import torch
from src.losses.localization import localization_loss
from src.losses.subevent_losses import diversity_loss

def test_hungarian_loss_finite():
 out={'pred_logits':torch.randn(2,4,2,requires_grad=True),'pred_spans':torch.rand(2,4,2,requires_grad=True)}
 targets=[{'spans':torch.tensor([[.2,.6]])},{'spans':torch.tensor([[.1,.8]])}]
 loss,_=localization_loss(out,targets); loss.backward(); assert torch.isfinite(loss)
def test_diversity_single_event_zero_grad():
 a=torch.softmax(torch.randn(1,1,5,requires_grad=True),-1); z=diversity_loss(a,torch.ones(1,1,dtype=torch.bool)); z.backward(); assert z.item()==0

def test_center_width_predictions_match_start_end_targets():
 out={'pred_logits':torch.tensor([[[0.,5.]]]),'pred_spans':torch.tensor([[[.5,.4]]])}
 total,parts=localization_loss(out,[{'spans':torch.tensor([[.3,.7]])}])
 assert parts['loss_l1'].item()<1e-6
 assert abs(parts['loss_giou'].item())<1e-6
