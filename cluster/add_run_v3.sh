#!/bin/bash
# Add a second training run inside an already running Slurm job (use with: srun --jobid=<id> --overlap).
# Usage: add_run.sh <run_name> <use_setg true|false> <num_workers> [config, default configs/default.yaml]
# Resumes from the last checkpoint, retries automatically, and on repeated failure waits up to
# HOLD_MIN minutes for ~/dac301_vmr/RETRY_<run_name> (touch it after a fix) instead of exiting.
RUN=$1; SETG=$2; NW=$3; CFG=${4:-configs/default.yaml}; AUTO_RETRIES=2; HOLD_MIN=120
ulimit -n $(ulimit -Hn)
source ~/dac301_vmr/venv/bin/activate
cd ~/dac301_vmr/HMNet_SETG
M=~/dac301_vmr/qvh_manifests/fixed; F=~/dac301_vmr/data_raw/qvhighlight; R=/home2/home/$USER/dac301_vmr/runs  # must match the original runs exactly (resume compares the config)
LOG=~/dac301_vmr/slurm_logs/${RUN}_added_${SLURM_JOB_ID}.log; attempt=0
while true; do
  rm -f $R/locks/$RUN.lock
  if ls $R/checkpoints/$RUN/epoch_*.pt >/dev/null 2>&1; then FRESH=""; else FRESH="--fresh"; fi
  echo "[$(date +%H:%M:%S)] $RUN attempt $attempt ${FRESH:-resume}" | tee -a $LOG
  python -u scripts/train.py --config $CFG --manifest $M/train.jsonl --val-manifest $M/val.jsonl --feature-root $F --run-name $RUN $FRESH \
    --set paths.drive_root=$R train.epochs=50 train.batch_size=32 train.grad_accum=1 train.num_workers=$NW model.use_setg=$SETG >> $LOG 2>&1
  code=$?
  if [ $code -eq 0 ]; then echo "[$(date +%H:%M:%S)] $RUN FINISHED OK" | tee -a $LOG; exit 0; fi
  attempt=$((attempt+1)); echo "[$(date +%H:%M:%S)] $RUN CRASHED (exit $code), see $LOG" | tee -a $LOG
  if [ $attempt -le $AUTO_RETRIES ]; then sleep 30; continue; fi
  echo "[$(date +%H:%M:%S)] $RUN NEEDS_FIX: holding up to $HOLD_MIN min. After fixing: touch ~/dac301_vmr/RETRY_$RUN" | tee -a $LOG
  rm -f ~/dac301_vmr/RETRY_$RUN; waited=0
  while [ ! -f ~/dac301_vmr/RETRY_$RUN ] && [ $waited -lt $HOLD_MIN ]; do sleep 60; waited=$((waited+1)); done
  if [ -f ~/dac301_vmr/RETRY_$RUN ]; then rm -f ~/dac301_vmr/RETRY_$RUN; attempt=0; continue; fi
  echo "[$(date +%H:%M:%S)] $RUN GAVE UP after hold" | tee -a $LOG; exit 1
done
