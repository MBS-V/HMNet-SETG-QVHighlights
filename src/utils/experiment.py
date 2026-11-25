"""Reproducibility manifest creation and updates."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,platform,sys,time
import torch
from omegaconf import OmegaConf
from .checkpoint import atomic_json_save

def file_sha256(path):
    p=Path(path)
    if not p.is_file(): return None
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def create_experiment_manifest(cfg,run_name,train_manifest,val_manifest,device,mamba_backend):
    return {'run_name':run_name,'project_version':'0.1.0','metric_definitions':'See docs/EVALUATION_PROTOCOL.md; validation checkpoint selection uses mean validation loss.','created_utc':datetime.now(timezone.utc).isoformat(),
            'model_variant':'HM-Net + SE-TG' if cfg.model.use_setg else 'HM-Net baseline',
            'dataset':cfg.data.dataset,'dataset_version':str(cfg.data.get('dataset_version','unspecified')),
            'feature_version':str(cfg.data.get('feature_version','unspecified')),
            'train_manifest':str(train_manifest),'train_manifest_sha256':file_sha256(train_manifest),
            'validation_manifest':str(val_manifest) if val_manifest else None,
            'validation_manifest_sha256':file_sha256(val_manifest) if val_manifest else None,
            'resolved_config':OmegaConf.to_container(cfg,resolve=True),
            'seed':int(cfg.seed),'device':str(device),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'python':sys.version,'platform':platform.platform(),'pytorch':torch.__version__,
            'cuda_runtime':torch.version.cuda,'mamba_backend':mamba_backend,
            'started_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':0.0,
            'last_epoch':None,'global_step':0,'best_metric':None,'latest_checkpoint':None,'best_checkpoint':None,'history':[]}

def save_experiment_manifest(manifest,path):
    atomic_json_save(manifest,path)
