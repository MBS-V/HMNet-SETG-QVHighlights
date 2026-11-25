import tempfile,torch
from pathlib import Path
from src.utils.checkpoint import CheckpointManager

def test_save_resume_prune_and_corruption_fallback():
 m=torch.nn.Linear(4,2); o=torch.optim.AdamW(m.parameters()); s=torch.optim.lr_scheduler.CosineAnnealingLR(o,10)
 with tempfile.TemporaryDirectory() as d:
  cm=CheckpointManager(d,keep_last=2)
  for ep in range(4): cm.save(ep,m,o,s,global_step=ep*3,best_metric=float(ep),is_best=ep==3)
  assert sorted(p.name for p in Path(d).glob('epoch_*.pt'))==['epoch_002.pt','epoch_003.pt']
  assert (Path(d)/'best.pt').exists(); r=cm.load_latest(m,o,s); assert r['start_epoch']==4 and r['global_step']==9
  (Path(d)/'epoch_003.pt').write_bytes(b'broken'); r=cm.load_latest(m,o,s); assert r['start_epoch']==3
