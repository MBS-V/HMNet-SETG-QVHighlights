#!/bin/bash
# Watcher + automatic memory guard for one running Slurm job (runs on the login node; very light).
# Usage: bash guard_v1.sh <jobid> <step_to_stop_if_memory_high, e.g. 10328.1> [limit_GB, default 29]
# Every 30 s: shows job state, epochs, problems and the job's real memory (from its cgroup, the same
# number Slurm uses). If memory passes the limit it cancels ONLY the given step, so the job and its
# GPU survive; the stopped runs keep their checkpoints and can be restarted with add_run_v3.sh.
set +H
JOB=$1; STEP=$2; LIMIT_GB=${3:-29}; L=~/dac301_vmr/slurm_logs; GUARD_LOG=$L/guard_$JOB.log; stopped=0
CG=$(find /sys/fs/cgroup -maxdepth 7 -type d -name "job_$JOB" 2>/dev/null | head -1)
mem_gb () {  # non-reclaimable memory of the job (anon), in GB; falls back to summed RSS of our python
  if [ -n "$CG" ] && [ -r "$CG/memory.stat" ]; then
    awk '$1=="anon"{printf "%.1f", $2/1e9; f=1} $1=="total_rss"{printf "%.1f", $2/1e9; f=1} END{if(!f) print "NA"}' "$CG/memory.stat"
  else
    ps -u $USER -o rss=,args= | awk '/scripts\/train.py/{s+=$1} END{printf "%.1f", s/1e6}'
  fi
}
while true; do
  clear; date
  if ! squeue -h -j $JOB -t R | grep -q .; then
    echo "##### JOB $JOB IS NOT RUNNING #####"; sacct -j $JOB -X -o State,ExitCode,Elapsed; printf '\a\a\a'; break
  fi
  squeue -u $USER -s | tail -n +2
  echo "epochs done: old-baseline $(grep -c '"epoch"' $L/hmnet_base_$JOB.out 2>/dev/null) | FIX baseline $(grep -c '"epoch"' $L/hmnet_baseline_fix_added_$JOB.log 2>/dev/null)  FIX se-tg $(grep -c '"epoch"' $L/hmnet_setg_fix_added_$JOB.log 2>/dev/null)"
  M=$(mem_gb); echo "job memory now: ${M} GB  (limit 32 GB, guard at ${LIMIT_GB} GB; source: ${CG:+cgroup}${CG:-ps-sum})"
  if [ "$M" != "NA" ] && [ $stopped -eq 0 ] && awk -v m="$M" -v l="$LIMIT_GB" 'BEGIN{exit !(m>l)}'; then
    msg="[$(date +%H:%M:%S)] MEMORY GUARD: ${M} GB > ${LIMIT_GB} GB -> stopping step $STEP only; job $JOB and its GPU are kept"
    echo "$msg" | tee -a $GUARD_LOG; scancel $STEP; stopped=1; printf '\a\a'
  fi
  [ -f $GUARD_LOG ] && { echo "### GUARD:"; tail -2 $GUARD_LOG; }
  E=$(grep -h "CRASHED\|NEEDS_FIX\|Traceback\|warn\]" $L/hmnet_base_$JOB.out $L/*fix_added_$JOB.log 2>/dev/null | tail -3)
  [ -n "$E" ] && { echo "### PROBLEM:"; echo "$E"; printf '\a'; }
  sleep 30
done
