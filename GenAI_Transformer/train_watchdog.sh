#!/usr/bin/env bash
# Auto-resume Transformer training when Vast GPU CUDA-crashes.
# Usage (after Restart instance):
#   cd ~/GenAi-Text-To-Music/GenAI_Transformer
#   bash train_watchdog.sh

set -u
source /venv/main/bin/activate
cd "$(dirname "$0")"

EPOCHS="${EPOCHS:-30}"
BS="${BS:-2}"
SEQ="${SEQ:-256}"
LR="${LR:-5e-5}"
LOG="${LOG:-$HOME/transformer_train.log}"
MAX_RETRIES="${MAX_RETRIES:-40}"

pick_resume() {
  local best="" best_n=-1
  shopt -s nullglob
  for f in checkpoints/checkpoint_epoch_*.pt; do
    n="${f##*_}"
    n="${n%.pt}"
    if [[ "$n" =~ ^[0-9]+$ ]] && (( n > best_n )); then
      best_n=$n
      best=$f
    fi
  done
  if [[ -f checkpoints/latest.pt ]]; then
    # prefer highest finished epoch checkpoint if newer than latest intent
    if (( best_n >= 1 )); then
      echo "$best"
      return
    fi
    echo "checkpoints/latest.pt"
    return
  fi
  if [[ -n "$best" ]]; then
    echo "$best"
    return
  fi
  if [[ -f checkpoints/best_model.pt ]]; then
    echo "checkpoints/best_model.pt"
    return
  fi
  echo ""
}

echo "[watchdog] log -> $LOG"
echo "[watchdog] target epochs=$EPOCHS batch=$BS seq=$SEQ lr=$LR"

for attempt in $(seq 1 "$MAX_RETRIES"); do
  # already done?
  if [[ -f checkpoints/checkpoint_epoch_${EPOCHS}.pt ]]; then
    echo "[watchdog] Found checkpoint_epoch_${EPOCHS}.pt — done."
    exit 0
  fi

  RESUME="$(pick_resume)"
  ARGS=(
    train.py
    --epochs "$EPOCHS"
    --batch_size "$BS"
    --max_seq_len "$SEQ"
    --lr "$LR"
    --no_early_stop
    --save_epochs "1,5,10,20,30"
    --no_cudnn
  )
  if [[ -n "$RESUME" ]]; then
    ARGS+=(--resume "$RESUME")
    echo "[watchdog] attempt $attempt/$MAX_RETRIES resume=$RESUME"
  else
    echo "[watchdog] attempt $attempt/$MAX_RETRIES fresh start"
  fi

  python "${ARGS[@]}" >>"$LOG" 2>&1
  code=$?
  echo "[watchdog] python exit=$code"

  if [[ $code -eq 0 ]]; then
    echo "[watchdog] training finished OK"
    exit 0
  fi

  echo "[watchdog] crash/CUDA — wait 5s then retry (Restart Vast UI if retries keep failing)"
  sleep 5
done

echo "[watchdog] gave up after $MAX_RETRIES retries"
exit 1
