#!/bin/bash
# Evaluate every finished run (best.pt) on QVHighlights val with the OFFICIAL Moment-DETR evaluator,
# copy training histories, and save demo output. Run inside the job:
#   srun --jobid=<job> --overlap bash ~/dac301_vmr/eval_all_v1.sh
# Results go to ~/dac301_vmr/results/ (small; download that folder).
source ~/dac301_vmr/venv/bin/activate
cd ~/dac301_vmr/HMNet_SETG
R=/home2/home/$USER/dac301_vmr/runs; M=~/dac301_vmr/qvh_manifests/fixed; F=~/dac301_vmr/data_raw/qvhighlight
OE=~/dac301_vmr/qvh_manifests/official_eval; OUT=~/dac301_vmr/results; mkdir -p $OUT/configs
# run name | base config | use_setg
RUNS="hmnet_baseline_e50:configs/default.yaml:false hmnet_setg_e50:configs/default.yaml:true hmnet_baseline_fix:configs/fixed.yaml:false hmnet_setg_fix:configs/fixed.yaml:true"
for spec in $RUNS; do
  IFS=: read RUN BASE SETG <<< "$spec"
  CK=$R/checkpoints/$RUN/best.pt
  [ -f $CK ] || { echo "== $RUN: no best.pt, skipped"; continue; }
  echo "== $RUN ($BASE, use_setg=$SETG)"
  CFG=$OUT/configs/$RUN.yaml; sed "s/use_setg: false/use_setg: $SETG/" $BASE > $CFG
  D=$OUT/$RUN; mkdir -p $D
  cp $R/checkpoints/$RUN/experiment_manifest.json $D/experiment_manifest.json
  python scripts/evaluate.py --config $CFG --manifest $M/val.jsonl --feature-root $F --checkpoint $CK --output-dir $D 2>&1 | grep -E "^loaded|wrote"
  ( cd $OE && PYTHONPATH=. python standalone_eval/eval.py --submission_path $D/predictions_official.jsonl \
      --gt_path $OE/data/highlight_val_release.jsonl --save_path $D/metrics_official.json > $D/official_eval_stdout.txt 2>&1 )
  python -c "import json;d=json.load(open('$D/metrics_official.json'))['brief'];print('   OFFICIAL:',{k:d[k] for k in d if k.startswith('MR-full')})"
  python scripts/demo_inference.py --config $CFG --checkpoint $CK --manifest $M/val.jsonl --feature-root $F --qids 2043 5266 --random 3 --seed 7 > $D/demo_output.txt 2>&1
done
cp ~/dac301_vmr/slurm_logs/*10328* $OUT/ 2>/dev/null
cd ~/dac301_vmr && python3 -m zipfile -c results.zip results && ls -la results.zip && echo EVAL_ALL_DONE
