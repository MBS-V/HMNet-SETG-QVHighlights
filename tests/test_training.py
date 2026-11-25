import torch
from torch.utils.data import DataLoader
from src.models.hmnet import HMNet
from src.data.collate import collate_moments
from src.engine.trainer import train_one_epoch,evaluate_loss

def test_one_batch_training_and_eval():
 rows=[]
 for i in range(2):
  rows.append({'video':torch.randn(5,12),'text':torch.randn(4,10),'query':'a b c d','qid':str(i),'video_id':'v','duration':10.,'span':torch.tensor([.1,.8]),'token_spans':torch.tensor([[0,4]]),'num_subevents':1,'parse':{},'saliency':None})
 dl=DataLoader(rows,batch_size=2,collate_fn=collate_moments)
 m=HMNet(video_dim=12,text_dim=10,hidden_dim=16,heads=4,state_dim=4,enhancement_layers=1,cofusion_layers=1,num_queries=3,decoder_layers=1,dropout=0)
 opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
 tr=train_one_epoch(m,dl,opt,torch.device('cpu'),0,grad_accum=3,amp=False)
 ev=evaluate_loss(m,dl,torch.device('cpu'))
 assert tr['steps']==1 and torch.isfinite(torch.tensor(tr['loss'])) and torch.isfinite(torch.tensor(ev['loss']))
