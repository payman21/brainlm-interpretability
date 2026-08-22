#!/bin/bash
# One Slurm array task = one subject: fetch 4 runs, parcellate, verify, delete.
# Idempotent: a subject whose 4 .npy files already exist is skipped, so the
# array can be resubmitted after any failure without redoing work.
#SBATCH --job-name=hcpya
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=${HCP_ROOT:?set HCP_ROOT to the cluster scratch directory}/logs/slurm-%A_%a.out

set -uo pipefail
ROOT=${HCP_ROOT:?set HCP_ROOT to the cluster scratch directory}
PY=${PYTHON:-python}
S5=$ROOT/scripts/s5cmd
ATLAS=$ROOT/atlas/A424+2mm.nii.gz
export AWS_PROFILE=hcp

# Subjects whose raw NIfTI is deliberately retained as a spot-check reserve.
KEEP_RAW_THROUGH=30

# Slurm caps array indices at MaxArraySize (1001 here), so the tail of the
# cohort is submitted as a second array with OFFSET set.
LINE=$((SLURM_ARRAY_TASK_ID + ${OFFSET:-0}))
SUBJ=$(sed -n "${LINE}p" $ROOT/hcp_subjects.txt)
[ -z "$SUBJ" ] && { echo "no subject at line $LINE"; exit 1; }

STAGE=$ROOT/staging/$SUBJ
OUT=$ROOT/parcels
STATUS=$ROOT/logs/status/$SUBJ.json
mkdir -p "$STAGE" "$OUT" "$ROOT/logs/status" "$ROOT/motion/$SUBJ"

RUNS="rfMRI_REST1_LR rfMRI_REST1_RL rfMRI_REST2_LR rfMRI_REST2_RL"

# Skip if already complete.
done_all=1
for R in $RUNS; do [ -f "$OUT/${SUBJ}_${R}.npy" ] || done_all=0; done
if [ $done_all -eq 1 ]; then
  echo "$SUBJ already complete, skipping"; rm -rf "$STAGE"; exit 0
fi

echo "=== $SUBJ (task $LINE) on $(hostname) at $(date -Is) ==="
fail=0

# --- fetch all four runs in parallel (4 streams per subject) -------------
for R in $RUNS; do
  [ -f "$OUT/${SUBJ}_${R}.npy" ] && continue
  KEY="s3://hcp-openaccess/HCP_1200/$SUBJ/MNINonLinear/Results/$R/${R}_hp2000_clean.nii.gz"
  $S5 cp "$KEY" "$STAGE/" >/dev/null 2>&1 &
done
wait

# Motion regressors: tiny, and the one thing not reconstructible from parcels.
for R in $RUNS; do
  $S5 cp "s3://hcp-openaccess/HCP_1200/$SUBJ/MNINonLinear/Results/$R/Movement_Regressors.txt" \
        "$ROOT/motion/$SUBJ/${R}_Movement_Regressors.txt" >/dev/null 2>&1 &
done
wait

# --- parcellate + verify, in parallel across the 4 allocated cores -------
for R in $RUNS; do
  [ -f "$OUT/${SUBJ}_${R}.npy" ] && continue
  NII="$STAGE/${R}_hp2000_clean.nii.gz"
  EXP=$(grep -m1 "/$SUBJ/MNINonLinear/Results/$R/" $ROOT/s3_manifest.jsonl | sed 's/.*"size":\([0-9]*\).*/\1/')
  ( $PY $ROOT/scripts/parcellate_run.py "$NII" "$ATLAS" "$OUT/${SUBJ}_${R}.npy" "$EXP" \
      > "$ROOT/logs/status/${SUBJ}_${R}.json" 2>&1 ) &
done
wait

# --- only delete voxels once every run is certified ----------------------
for R in $RUNS; do
  # The .npy is written by atomic rename only after the read-back check, so its
  # existence IS the certification. Checking the status file instead falsely
  # fails any resumed subject whose runs were skipped on this pass.
  if [ -f "$OUT/${SUBJ}_${R}.npy" ]; then
    if [ "$LINE" -gt "$KEEP_RAW_THROUGH" ]; then
      rm -f "$STAGE/${R}_hp2000_clean.nii.gz"
    fi
  else
    echo "VERIFY FAILED $SUBJ $R:"; cat "$ROOT/logs/status/${SUBJ}_${R}.json" 2>/dev/null
    fail=1
  fi
done

if [ $fail -eq 0 ]; then
  [ "$LINE" -gt "$KEEP_RAW_THROUGH" ] && rmdir "$STAGE" 2>/dev/null
  echo "{\"subject\":\"$SUBJ\",\"ok\":true,\"line\":$LINE,\"host\":\"$(hostname)\",\"done\":\"$(date -Is)\"}" > "$STATUS"
  echo "=== $SUBJ OK ==="
else
  echo "{\"subject\":\"$SUBJ\",\"ok\":false,\"line\":$LINE}" > "$STATUS"
  echo "=== $SUBJ FAILED (raw kept at $STAGE) ==="
  exit 1
fi
