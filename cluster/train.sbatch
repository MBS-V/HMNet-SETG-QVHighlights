#!/bin/bash
# Single-model job used for the first MFS runs (job 10328 ran the original-settings baseline with this).
# Submit: sbatch --job-name=hmnet_base --export=ALL,RUN=hmnet_baseline_e50,SETG=false cluster/train.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home2/home/%u/dac301_vmr/slurm_logs/%x_%j.out
set -e
ulimit -n $(ulimit -Hn)
source ~/dac301_vmr/venv/bin/activate
cd ~/dac301_vmr/HMNet_SETG
echo "RUN=$RUN SETG=$SETG host=$(hostname) start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(); print('cuda OK', torch.cuda.get_device_name(0))"
python -m pytest -q 2>&1 | tail -1 | tee /dev/stderr | grep -q "46 passed"
M=~/dac301_vmr/qvh_manifests/fixed; F=~/dac301_vmr/data_raw/qvhighlight; R=/home2/home/$USER/dac301_vmr/runs
if ls $R/checkpoints/$RUN/epoch_*.pt >/dev/null 2>&1; then FRESH=""; echo "RESUMING $RUN"; else FRESH="--fresh"; fi
python -u scripts/train.py --manifest $M/train.jsonl --val-manifest $M/val.jsonl --feature-root $F --run-name $RUN $FRESH \
  --set paths.drive_root=$R train.epochs=50 train.batch_size=32 train.grad_accum=1 train.num_workers=6 model.use_setg=$SETG
echo "end=$(date)"
