#!/bin/bash
# Progress report for the HCP-YA fetch. Completion is judged from parcels/ on
# disk, not from status files -- the .npy is only written after the read-back
# check, so its presence is the certification.
#   ./status.sh             report
#   ./status.sh --resubmit  requeue every incomplete subject
export SLURM_CONF=/etc/slurm/slurm.conf
ROOT=${HCP_ROOT:?set HCP_ROOT to the cluster scratch directory}
cd $ROOT || exit 1
RUNS="rfMRI_REST1_LR rfMRI_REST1_RL rfMRI_REST2_LR rfMRI_REST2_RL"

total=$(wc -l < hcp_subjects.txt)
done_runs=$(ls parcels/*.npy 2>/dev/null | wc -l)
> /tmp/incomplete.lines
n=0
while read s; do
  n=$((n+1)); c=0
  for r in $RUNS; do [ -f "parcels/${s}_${r}.npy" ] && c=$((c+1)); done
  [ $c -lt 4 ] && echo "$n $s $c" >> /tmp/incomplete.lines
done < hcp_subjects.txt
inc=$(wc -l < /tmp/incomplete.lines)

echo "runs      : $done_runs / $((total*4))  ($((done_runs*100/(total*4)))%)"
echo "subjects  : $((total-inc)) / $total complete"
echo "queued    : $(squeue -u $USER -h | wc -l) tasks still in Slurm"
echo "parcels   : $(du -sh parcels 2>/dev/null | cut -f1)   motion: $(du -sh motion 2>/dev/null | cut -f1)   staging: $(du -sh staging 2>/dev/null | cut -f1)"
echo "dropout   : $(grep -ho '"zero_parcels": \[[0-9][^]]*\]' logs/status/*_rfMRI*.json 2>/dev/null | wc -l) runs with >=1 empty parcel"
df -h /dataset | tail -1

if [ "$inc" -gt 0 ]; then
  echo; echo "incomplete subjects ($inc), first 10:"; head -10 /tmp/incomplete.lines | sed 's/^/  line /'
  if [ "${1:-}" = "--resubmit" ]; then
    # Skip anything Slurm is already working on.
    if [ "$(squeue -u $USER -h | wc -l)" -gt 0 ]; then
      echo "jobs still queued -- wait for them to finish before resubmitting"; exit 0
    fi
    main=$(awk '$1<=1000{printf "%s,",$1}' /tmp/incomplete.lines | sed 's/,$//')
    tail_=$(awk '$1>1000{printf "%s,",$1-1000}' /tmp/incomplete.lines | sed 's/,$//')
    [ -n "$main" ] && echo "resubmitted main: $(sbatch --parsable --array=${main}%16 scripts/process_subject.sh)"
    [ -n "$tail_" ] && echo "resubmitted tail: $(sbatch --parsable --array=${tail_}%4 --export=ALL,OFFSET=1000 scripts/process_subject.sh)"
  else
    echo "  run './status.sh --resubmit' once the queue is empty to requeue these"
  fi
else
  echo; echo "ALL COMPLETE."
fi
