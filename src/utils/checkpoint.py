"""Atomic checkpoint manager with RNG capture, corruption fallback and retention."""
import json, os, random, tempfile
from pathlib import Path
import numpy as np
import torch

def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'].cpu())
    if s.get('cuda') is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])

def atomic_torch_save(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent); os.close(fd)
    try:
        torch.save(obj,tmp)
        with open(tmp,'rb') as f: os.fsync(f.fileno())
        os.replace(tmp,path)
        # Verify readability before reporting success.
        torch.load(path,map_location='cpu',weights_only=False)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def atomic_json_save(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent); os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(obj,indent=2),encoding='utf-8')
        with open(tmp,'rb') as f: os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

class CheckpointManager:
    def __init__(self,directory,keep_last=2):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True); self.keep_last=max(1,int(keep_last))
    def save(self,epoch,model,optimizer=None,scheduler=None,scaler=None,global_step=0,best_metric=None,metadata=None,is_best=False):
        state={'epoch':int(epoch),'global_step':int(global_step),'model':model.state_dict(),
               'optimizer':optimizer.state_dict() if optimizer else None,'scheduler':scheduler.state_dict() if scheduler else None,
               'scaler':scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
               'best_metric':best_metric,'metadata':metadata or {},'rng':capture_rng()}
        path=self.directory/f'epoch_{epoch:03d}.pt'; atomic_torch_save(state,path)
        atomic_json_save({'file':path.name,'epoch':int(epoch)},self.directory/'latest.json')
        if is_best: atomic_torch_save({'epoch':int(epoch),'model':model.state_dict(),'best_metric':best_metric,'metadata':metadata or {}},self.directory/'best.pt')
        self._prune(); return path
    def _prune(self):
        files=sorted(self.directory.glob('epoch_*.pt'))
        for p in files[:-self.keep_last]: p.unlink(missing_ok=True)
    def load_latest(self,model,optimizer=None,scheduler=None,scaler=None,restore_random=True):
        for path in sorted(self.directory.glob('epoch_*.pt'),reverse=True):
            try:
                state=torch.load(path,map_location='cpu',weights_only=False)
                required={'epoch','model','rng'}
                if not required.issubset(state): raise ValueError('checkpoint missing required keys')
                model.load_state_dict(state['model'])
                if optimizer is not None and state.get('optimizer') is not None: optimizer.load_state_dict(state['optimizer'])
                if scheduler is not None and state.get('scheduler') is not None: scheduler.load_state_dict(state['scheduler'])
                if scaler is not None and state.get('scaler') is not None: scaler.load_state_dict(state['scaler'])
                if restore_random: restore_rng(state['rng'])
                return {'start_epoch':int(state['epoch'])+1,'epoch':int(state['epoch']),'global_step':int(state.get('global_step',0)),
                        'best_metric':state.get('best_metric'),'metadata':state.get('metadata',{}),'path':str(path)}
            except Exception as exc:
                print(f'[checkpoint] skipping invalid {path.name}: {exc}')
        return None
