#!/bin/bash
#SBATCH --job-name=hmnet_both
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home2/home/%u/dac301_vmr/slurm_logs/%x_%j.out
# Runs the baseline and SE-TG side by side on one GPU.
# Each run resumes from its last checkpoint, retries automatically, and on repeated failure
# keeps the GPU for up to HOLD_MIN minutes waiting for ~/dac301_vmr/RETRY_<run> (touch it after a fix).
ulimit -n $(ulimit -Hn)
source ~/dac301_vmr/venv/bin/activate
cd ~/dac301_vmr/HMNet_SETG
echo "host=$(hostname) start=$(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(); print('cuda OK', torch.cuda.get_device_name(0))" || exit 1
python -m pytest -q 2>&1 | tail -1 | tee /dev/stderr | grep -q "46 passed" || { echo "TESTS FAILED"; exit 1; }
M=~/dac301_vmr/qvh_manifests/fixed; F=~/dac301_vmr/data_raw/qvhighlight; R=/home2/home/$USER/dac301_vmr/runs
LOGS=~/dac301_vmr/slurm_logs; AUTO_RETRIES=2; HOLD_MIN=120

run_one () {  # $1=run name  $2=use_setg  $3=num_workers
  local RUN=$1 SETG=$2 NW=$3 attempt=0 LOG=$LOGS/${1}_${SLURM_JOB_ID}.log
  while true; do
    rm -f $R/locks/$RUN.lock   # only this job uses this run name; clears a stale lock left by a killed process
    if ls $R/checkpoints/$RUN/epoch_*.pt >/dev/null 2>&1; then FRESH=""; else FRESH="--fresh"; fi
    echo "[$(date +%H:%M:%S)] $RUN attempt $attempt ${FRESH:-resume}" | tee -a $LOG
    python -u scripts/train.py --manifest $M/train.jsonl --val-manifest $M/val.jsonl --feature-root $F --run-name $RUN $FRESH \
      --set paths.drive_root=$R train.epochs=50 train.batch_size=32 train.grad_accum=1 train.num_workers=$NW model.use_setg=$SETG >> $LOG 2>&1
    code=$?
    if [ $code -eq 0 ]; then echo "[$(date +%H:%M:%S)] $RUN FINISHED OK" | tee -a $LOG; return 0; fi
    attempt=$((attempt+1)); echo "[$(date +%H:%M:%S)] $RUN CRASHED (exit $code), see $LOG" | tee -a $LOG
    if [ $attempt -le $AUTO_RETRIES ]; then sleep 30; continue; fi
    echo "[$(date +%H:%M:%S)] $RUN NEEDS_FIX: holding GPU up to $HOLD_MIN min. After fixing: touch ~/dac301_vmr/RETRY_$RUN" | tee -a $LOG
    rm -f ~/dac301_vmr/RETRY_$RUN; waited=0
    while [ ! -f ~/dac301_vmr/RETRY_$RUN ] && [ $waited -lt $HOLD_MIN ]; do sleep 60; waited=$((waited+1)); done
    if [ -f ~/dac301_vmr/RETRY_$RUN ]; then rm -f ~/dac301_vmr/RETRY_$RUN; attempt=0; continue; fi
    echo "[$(date +%H:%M:%S)] $RUN GAVE UP after hold" | tee -a $LOG; return 1
  done
}

run_one hmnet_baseline_e50 false 2 &
run_one hmnet_setg_e50 true 6 &
wait
echo "all done end=$(date)"
