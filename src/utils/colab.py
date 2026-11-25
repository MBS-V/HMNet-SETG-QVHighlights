"""Colab helpers and advisory run lock. Drive locks are not strongly consistent."""
import json, os, time, socket
from pathlib import Path
from omegaconf import OmegaConf

def in_colab():
    try: import google.colab; return True
    except ImportError: return False

def mount_drive():
    if in_colab():
        from google.colab import drive
        drive.mount('/content/drive')

def flush_drive():
    if in_colab():
        from google.colab import drive
        drive.flush_and_unmount()

def load_config(path='configs/default.yaml',drive_root=None,**overrides):
    path=Path(path)
    if path.name!='default.yaml' and (path.parent/'default.yaml').exists():
        cfg=OmegaConf.merge(OmegaConf.load(path.parent/'default.yaml'),OmegaConf.load(path))
    else: cfg=OmegaConf.load(path)
    if drive_root: cfg.paths.drive_root=drive_root
    for k,v in overrides.items(): OmegaConf.update(cfg,k.replace('__','.'),v,merge=True)
    OmegaConf.resolve(cfg)
    return cfg

class RunLock:
    def __init__(self,path,who=None,stale_after=21600):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.who=who or f'{socket.gethostname()}:{os.getpid()}'; self.stale_after=stale_after
    def _read(self):
        try: return json.loads(self.path.read_text())
        except Exception: return None
    def acquire(self,force=False):
        cur=self._read()
        if cur and cur.get('owner')!=self.who and time.time()-cur.get('timestamp',0)<self.stale_after and not force:
            raise RuntimeError(f"Run locked by {cur.get('owner')}; heartbeat age={int(time.time()-cur.get('timestamp',0))}s")
        payload={'owner':self.who,'timestamp':time.time()}
        tmp=self.path.with_suffix(self.path.suffix+'.tmp'); tmp.write_text(json.dumps(payload)); os.replace(tmp,self.path)
    def heartbeat(self):
        cur=self._read()
        if not cur or cur.get('owner')!=self.who: raise RuntimeError('Cannot heartbeat: lock is not owned by this process')
        tmp=self.path.with_suffix(self.path.suffix+'.tmp'); tmp.write_text(json.dumps({'owner':self.who,'timestamp':time.time()})); os.replace(tmp,self.path)
    def release(self):
        cur=self._read()
        if cur and cur.get('owner')==self.who: self.path.unlink(missing_ok=True)
