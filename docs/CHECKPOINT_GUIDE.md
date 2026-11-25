# Checkpoint and runtime handoff

`src/utils/checkpoint.py` writes `epoch_XXX.pt` atomically through a temporary file and `os.replace`, verifies readability, records model/optimizer/scheduler/scaler state, epoch, global step, best metric, metadata and Python/NumPy/PyTorch/CUDA RNG states. It retains the newest two epoch files by default and maintains a separate `best.pt` weights snapshot. Loading scans newest-to-oldest and skips unreadable/corrupt epoch files.

## Resume
Use the same config and `--run-name` and omit `--fresh`:
```bash
python scripts/train.py --config configs/qvhighlights.yaml --manifest /path/train.jsonl --feature-root /path/features --run-name qvh_hmnet_baseline
```
The script calls `load_latest` before the epoch loop. The next epoch starts at saved epoch + 1. The run's `experiment_manifest.json` records the resolved config, manifest hashes, hardware, software versions, Mamba backend, checkpoint paths and epoch history. Resume is rejected if the resolved config or train/validation manifest hash differs.

## Verify a checkpoint
```bash
python scripts/verify_checkpoint.py /path/to/checkpoints/run/epoch_001.pt
```

## Safe handoff
Stop the active training process between epochs, confirm the latest checkpoint and `latest.json` are visible in Drive, The CLI releases its run lock automatically on normal exit. The project lock is advisory because Drive synchronization may be delayed; do not assume it provides transactional multi-writer safety. The training CLI acquires a run-specific advisory lock under `paths.locks`, refreshes it after each epoch and releases it on normal exit. Stale locks are considered after six hours. Drive synchronization is not strongly consistent, so use unique run names and do not treat this as a transactional distributed lock.
