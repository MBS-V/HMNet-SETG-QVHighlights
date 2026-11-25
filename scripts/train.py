"""Configuration-driven resumable training CLI."""
import argparse, json, sys, time
import yaml
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.utils.device import get_device
from src.data.dataset import FeatureMomentDataset
from src.data.collate import collate_moments
from src.models.hmnet import HMNet
from src.models.mamba_block import MAMBA_SSM_AVAILABLE
from src.engine.optimizer import build_optimizer
from src.engine.scheduler import build_scheduler
from src.engine.trainer import train_one_epoch,evaluate_loss,evaluate_metrics
from src.utils.checkpoint import CheckpointManager
from src.utils.colab import RunLock
from src.utils.experiment import create_experiment_manifest,save_experiment_manifest,file_sha256

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/default.yaml')
    p.add_argument('--manifest')
    p.add_argument('--val-manifest')
    p.add_argument('--feature-root')
    p.add_argument('--run-name',default='hmnet_run')
    p.add_argument('--fresh',action='store_true',help='Start from scratch; use a new run name to preserve old checkpoints')
    p.add_argument('--set',nargs='*',default=[],help='config overrides, e.g. train.epochs=1 train.batch_size=32')
    a=p.parse_args()
    overrides={}
    for kv in a.set:
        k,v=kv.split('=',1); overrides[k]=yaml.safe_load(v)

    cfg=load_config(a.config,overrides); seed_everything(cfg.seed); dev=get_device(cfg.device)
    manifest=a.manifest or cfg.data.manifest; root=a.feature_root or cfg.data.feature_dir
    ds=FeatureMomentDataset(manifest,root,max_subevents=cfg.data.max_subevents,l2_normalize=bool(cfg.data.get('l2_normalize',False)))
    collate=lambda x:collate_moments(x,cfg.data.max_subevents)
    loader=DataLoader(ds,batch_size=cfg.train.batch_size,shuffle=True,num_workers=cfg.train.num_workers,collate_fn=collate)
    val_loader=None
    if a.val_manifest:
        val_ds=FeatureMomentDataset(a.val_manifest,root,max_subevents=cfg.data.max_subevents,l2_normalize=bool(cfg.data.get('l2_normalize',False)))
        val_loader=DataLoader(val_ds,batch_size=cfg.train.batch_size,shuffle=False,num_workers=cfg.train.num_workers,collate_fn=collate)

    model=HMNet(video_dim=cfg.data.video_dim,text_dim=cfg.data.text_dim,hidden_dim=cfg.model.hidden_dim,heads=cfg.model.heads,state_dim=cfg.model.state_dim,
        enhancement_layers=cfg.model.enhancement_layers,cofusion_layers=cfg.model.cofusion_layers,num_queries=cfg.model.num_queries,decoder_layers=cfg.model.decoder_layers,
        max_seq_len=cfg.model.max_seq_len,dropout=cfg.model.dropout,drop_path=cfg.model.drop_path,use_setg=cfg.model.use_setg,max_subevents=cfg.data.max_subevents,
        use_reconstruction=cfg.model.use_reconstruction,vocab_size=cfg.data.vocab_size,reconstruction_mask_fraction=cfg.reconstruction.mask_fraction,
        reconstruction_temperature=cfg.reconstruction.temperature,reconstruction_positive_iou=cfg.reconstruction.positive_iou).to(dev)
    opt=build_optimizer(model,cfg.train.lr,cfg.train.weight_decay)
    sched=build_scheduler(opt,cfg.train.warmup_epochs,cfg.train.epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=(cfg.train.amp and dev.type=='cuda'))
    ckpt_dir=Path(cfg.paths.checkpoints)/a.run_name
    ckpt=CheckpointManager(ckpt_dir,cfg.train.keep_last_checkpoints)
    existing=list(ckpt_dir.glob('epoch_*.pt'))
    has_existing=bool(existing) or (ckpt_dir/'best.pt').exists()
    if a.fresh and has_existing:
        raise RuntimeError('This run name already has checkpoints. Choose a new --run-name for a fresh experiment; existing checkpoints are preserved.')

    backend='official_mamba_ssm' if MAMBA_SSM_AVAILABLE and dev.type=='cuda' else 'pytorch_selective_scan_fallback'
    manifest_path=ckpt_dir/'experiment_manifest.json'
    if manifest_path.exists() and not a.fresh:
        saved_backend=json.loads(manifest_path.read_text(encoding='utf-8')).get('mamba_backend')
        if saved_backend and saved_backend!=backend:
            raise RuntimeError(f'This run was started with Mamba backend {saved_backend!r} but this session has {backend!r}. '
                               'Their weights are not interchangeable. Install/uninstall mamba-ssm to match, or use a new --run-name.')
    select_metric=cfg.train.get('select_metric','loss')
    higher_is_better=select_metric!='loss'
    if higher_is_better and val_loader is None:
        raise RuntimeError(f'train.select_metric={select_metric} needs --val-manifest')
    lock=RunLock(Path(cfg.paths.locks)/(a.run_name+'.lock'))
    lock.acquire()
    try:
        state=None if a.fresh else ckpt.load_latest(model,opt,sched,scaler)
        if state is None and has_existing and not a.fresh:
            raise RuntimeError('Checkpoint files exist but none are valid. Inspect them or choose a new run name; refusing to overwrite them.')
        if manifest_path.exists() and state is not None:
            experiment=json.loads(manifest_path.read_text(encoding='utf-8'))
            current=OmegaConf.to_container(cfg,resolve=True)
            if experiment.get('resolved_config')!=current:
                raise RuntimeError('Resolved config differs from the saved experiment manifest. Use the original config or a new run name.')
            if experiment.get('train_manifest_sha256')!=file_sha256(manifest):
                raise RuntimeError('Training manifest changed since the saved run. Use a new run name rather than resuming on different data.')
            expected_val_hash=file_sha256(a.val_manifest) if a.val_manifest else None
            if experiment.get('validation_manifest_sha256')!=expected_val_hash:
                raise RuntimeError('Validation manifest changed since the saved run. Use a new run name.')
        else:
            experiment=create_experiment_manifest(cfg,a.run_name,manifest,a.val_manifest,dev,backend)
        start=state['start_epoch'] if state else 0
        default_best=float('-inf') if higher_is_better else float('inf')
        best=state['best_metric'] if state and state['best_metric'] is not None else default_best
        step=state['global_step'] if state else 0
        elapsed_before=float(experiment.get('elapsed_seconds',0.0)); run_start=time.time()

        for epoch in range(start,cfg.train.epochs):
            tr=train_one_epoch(model,loader,opt,dev,epoch,scaler,cfg.train.grad_accum,cfg.train.grad_clip,cfg.train.amp,dict(cfg.loss))
            val=evaluate_loss(model,val_loader,dev,dict(cfg.loss)) if val_loader is not None else None
            if higher_is_better:
                val_metrics=evaluate_metrics(model,val_loader,dev); val.update({'metrics':val_metrics})
                metric=float(val_metrics[select_metric])
            else:
                metric=val['loss'] if val is not None else tr['loss']
            sched.step()
            is_best=metric>best if higher_is_better else metric<best
            if is_best: best=metric
            step+=tr['steps']
            path=ckpt.save(epoch,model,opt,sched,scaler,step,best,{'config':a.config,'dataset':cfg.data.dataset,'run_name':a.run_name},is_best)
            experiment['last_epoch']=epoch; experiment['global_step']=step; experiment['best_metric']=best
            experiment['latest_checkpoint']=str(path)
            experiment['best_checkpoint']=str(ckpt_dir/'best.pt') if (ckpt_dir/'best.pt').exists() else None
            experiment['elapsed_seconds']=elapsed_before+(time.time()-run_start)
            experiment.setdefault('history',[]).append({'epoch':epoch,'train':tr,'validation':val,
                'selection_metric':select_metric if higher_is_better else ('validation_loss' if val is not None else 'training_loss'),'selected_metric':metric})
            save_experiment_manifest(experiment,manifest_path)
            lock.heartbeat()
            print(json.dumps({'epoch':epoch,'train':tr,'validation':val,
                'selection_metric':select_metric if higher_is_better else ('validation_loss' if val is not None else 'training_loss'),'best':best,
                'checkpoint':str(path),'experiment_manifest':str(manifest_path)}))
    finally:
        lock.release()

if __name__=='__main__': main()
