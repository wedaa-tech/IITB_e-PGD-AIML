#!/usr/bin/env bash
# run_all.sh — end-to-end pipeline for the rnn_sentiment (IMDB) project.
#
# Runs the seven project scripts in the right order:
#   1.  main.py          — baseline RNN / LSTM / Attention with a learned
#                          128-dim embedding (B3, B5, B6 baseline)
#   2.  experiment.py    — {RNN, LSTM, Attention} × {Word2Vec, GloVe}
#                          (B4 + B5 pretrained-embedding comparison)
#   3.  run_hparam.py    — L × h sweep for RNN and LSTM  (B3 hyperparameter study)
#   4.  analyze.py       — plots + classification reports for step 1 checkpoints
#   5.  analyze_b4.py    — B4 plots (embedding comparison, training curves)
#   6.  analyze_b5.py    — B5 plots (RNN vs LSTM, stability, convergence)
#   7.  analyze_b6.py    — B6 plots (attention heatmaps, overhead, entropy)
#
# Usage:
#   ./run_all.sh                        # run every step with defaults (10 ep)
#   ./run_all.sh --epochs 5             # override the epoch count
#   ./run_all.sh --from 4               # resume at step 4 (analysis only)
#   ./run_all.sh --only 3               # just run the hparam sweep
#   ./run_all.sh --only 5,6,7           # run analysis scripts only
#   ./run_all.sh --from 2 --to 3 -e 6   # training steps 2 and 3, 6 epochs each
#   ./run_all.sh --dry-run              # print the plan without executing
#   ./run_all.sh --list                 # list every step and exit
#
# Notes:
#   * Matplotlib backend is forced to "Agg" so figures are saved to disk
#     instead of popping up and blocking the pipeline.
#   * PyTorch MPS fallback is enabled for Apple Silicon.
#   * The epoch count is propagated via the RNN_EPOCHS env var; src/config.py
#     reads it. experiment.py and src/hparam_search.py then inherit from
#     src.config so one flag controls all three training scripts.
#   * Exits non-zero (and stops the pipeline) on the first failure.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Non-blocking plots + MPS fallback for ops not yet implemented on Apple GPU.
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

# Step definitions: "label|script"
STEPS=(
  "baseline models (RNN/LSTM/Attention, learned emb)    |main.py"
  "embedding grid (3 models x {word2vec, glove})        |experiment.py"
  "hyperparameter sweep (L x h, RNN + LSTM)             |run_hparam.py"
  "analysis: baseline plots + reports                   |analyze.py"
  "analysis: B4 embedding comparison                    |analyze_b4.py"
  "analysis: B5 RNN vs LSTM                             |analyze_b5.py"
  "analysis: B6 attention study                         |analyze_b6.py"
)

# Analysis scripts never train — they're safe to re-run and don't need
# the RNN_EPOCHS override to mean anything.
ANALYSIS_SCRIPTS=("analyze.py" "analyze_b4.py" "analyze_b5.py" "analyze_b6.py")

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS=""       # empty = use NUM_EPOCHS default from src/config.py (10)
FROM_STEP=1
TO_STEP=${#STEPS[@]}
ONLY_STEPS=""
DRY_RUN=0
DO_LIST=0

print_help () {
  sed -n '2,34p' "$0"
  echo
  echo "Flags:"
  echo "  -e, --epochs N        Train for N epochs (sets RNN_EPOCHS=N)"
  echo "  --from K              Start at step K (1-based)"
  echo "  --to K                Stop after step K"
  echo "  --only LIST           Comma-separated step numbers (e.g. 1,3,5)"
  echo "  --list                List steps and exit"
  echo "  --dry-run             Print the plan, don't execute"
  echo "  -h, --help            Show this message"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--epochs)
      EPOCHS="${2:?--epochs requires a positive integer}"
      shift 2
      ;;
    --epochs=*)
      EPOCHS="${1#*=}"
      shift
      ;;
    --from)
      FROM_STEP="${2:?--from requires a step number}"
      shift 2
      ;;
    --to)
      TO_STEP="${2:?--to requires a step number}"
      shift 2
      ;;
    --only)
      ONLY_STEPS="${2:?--only requires a comma-separated list}"
      shift 2
      ;;
    --only=*)
      ONLY_STEPS="${1#*=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --list)
      DO_LIST=1
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Try: $0 --help" >&2
      exit 2
      ;;
  esac
done

# Validate epoch count.
if [[ -n "$EPOCHS" ]]; then
  if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --epochs must be a positive integer (got: $EPOCHS)" >&2
    exit 2
  fi
  export RNN_EPOCHS="$EPOCHS"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step selection
# ─────────────────────────────────────────────────────────────────────────────

total=${#STEPS[@]}

# Resolve --only into a concrete, sorted, unique list if provided.
selected_steps=()
if [[ -n "$ONLY_STEPS" ]]; then
  IFS=',' read -r -a raw_only <<< "$ONLY_STEPS"
  for s in "${raw_only[@]}"; do
    s="${s// /}"
    if ! [[ "$s" =~ ^[0-9]+$ ]] || (( s < 1 || s > total )); then
      echo "Error: --only entry '$s' is not a valid step (1..$total)" >&2
      exit 2
    fi
    selected_steps+=("$s")
  done
  # Deduplicate + sort.
  IFS=$'\n' selected_steps=($(printf "%s\n" "${selected_steps[@]}" | sort -nu))
  unset IFS
else
  if ! [[ "$FROM_STEP" =~ ^[0-9]+$ ]] || (( FROM_STEP < 1 || FROM_STEP > total )); then
    echo "Error: --from must be between 1 and $total (got: $FROM_STEP)" >&2
    exit 2
  fi
  if ! [[ "$TO_STEP" =~ ^[0-9]+$ ]] || (( TO_STEP < FROM_STEP || TO_STEP > total )); then
    echo "Error: --to must be between $FROM_STEP and $total (got: $TO_STEP)" >&2
    exit 2
  fi
  for ((i = FROM_STEP; i <= TO_STEP; i++)); do
    selected_steps+=("$i")
  done
fi

# Helper: extract label / script for a 1-based step number.
step_label  () { local idx=$(( $1 - 1 )); echo "${STEPS[$idx]%%|*}"           | sed -e 's/[[:space:]]*$//'; }
step_script () { local idx=$(( $1 - 1 )); echo "${STEPS[$idx]##*|}"; }

# ─────────────────────────────────────────────────────────────────────────────
# --list
# ─────────────────────────────────────────────────────────────────────────────

if [[ "$DO_LIST" -eq 1 ]]; then
  echo "rnn_sentiment pipeline — ${total} steps"
  echo "────────────────────────────────────────────────────────────────"
  for ((i = 1; i <= total; i++)); do
    printf "  %d. %-54s  (%s)\n" "$i" "$(step_label "$i")" "$(step_script "$i")"
  done
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Environment auto-activation
# ─────────────────────────────────────────────────────────────────────────────

# Prefer the rnn_sentiment conda env if available (the project was originally
# built against miniforge3/envs/rnn_sentiment), else fall back to a local venv,
# else just use whatever "python" is on PATH.
activate_env () {
  if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" == "rnn_sentiment" ]]; then
    return 0
  fi
  # Try conda first.
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "rnn_sentiment"; then
      conda activate rnn_sentiment
      return 0
    fi
  fi
  # Fall back to a local venv if the user created one.
  if [[ -f "venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "venv/bin/activate"
    return 0
  fi
  return 0  # use whatever python is on PATH
}

activate_env

PYTHON_BIN="${PYTHON:-python}"

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════════"
echo "  rnn_sentiment — end-to-end pipeline"
echo "════════════════════════════════════════════════════════════════════"
echo "  Python        : $("$PYTHON_BIN" -V 2>&1)"
echo "  Working dir   : $SCRIPT_DIR"
echo "  MPLBACKEND    : $MPLBACKEND"
echo "  MPS fallback  : $PYTORCH_ENABLE_MPS_FALLBACK"
if [[ -n "${RNN_EPOCHS:-}" ]]; then
  echo "  Epochs (env)  : $RNN_EPOCHS  (RNN_EPOCHS is exported)"
else
  echo "  Epochs        : default from src/config.py (NUM_EPOCHS=10)"
fi
echo "  Steps to run  : ${selected_steps[*]}"
echo "  Dry run       : $([[ $DRY_RUN -eq 1 ]] && echo yes || echo no)"
echo "════════════════════════════════════════════════════════════════════"

start_wall=$(date +%s)
executed=()
skipped=()

for step in "${selected_steps[@]}"; do
  label=$(step_label "$step")
  script=$(step_script "$step")

  echo
  echo "────────────────────────────────────────────────────────────────────"
  echo "  Step $step / $total — $label"
  echo "  Script        : $script"
  echo "────────────────────────────────────────────────────────────────────"

  if [[ ! -f "$script" ]]; then
    echo "  [run_all] Script '$script' not found — skipping."
    skipped+=("$step:$script")
    continue
  fi

  cmd=("$PYTHON_BIN" "$script")
  echo "  Command       : ${cmd[*]}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    skipped+=("$step:$script (dry-run)")
    continue
  fi

  step_start=$(date +%s)
  if "${cmd[@]}"; then
    step_elapsed=$(( $(date +%s) - step_start ))
    printf "  [run_all] Step %d OK  (%dm %ds)\n" "$step" $((step_elapsed / 60)) $((step_elapsed % 60))
    executed+=("$step:$script")
  else
    rc=$?
    step_elapsed=$(( $(date +%s) - step_start ))
    echo "  [run_all] Step $step FAILED after ${step_elapsed}s — stopping pipeline."
    # Don't bail out of the summary; fall through to the end block.
    end_wall=$(date +%s)
    total_elapsed=$(( end_wall - start_wall ))
    echo
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Pipeline summary (FAILED)"
    echo "════════════════════════════════════════════════════════════════════"
    printf "  Total time    : %dm %ds\n" $((total_elapsed / 60)) $((total_elapsed % 60))
    echo  "  Executed      : ${executed[*]:-none}"
    echo  "  Skipped       : ${skipped[*]:-none}"
    echo  "  Failed step   : $step ($script)"
    exit "$rc"
  fi
done

end_wall=$(date +%s)
total_elapsed=$(( end_wall - start_wall ))

echo
echo "════════════════════════════════════════════════════════════════════"
echo "  Pipeline summary"
echo "════════════════════════════════════════════════════════════════════"
printf "  Total time    : %dm %ds\n" $((total_elapsed / 60)) $((total_elapsed % 60))
echo  "  Executed      : ${executed[*]:-none}"
echo  "  Skipped       : ${skipped[*]:-none}"
echo
echo "  Outputs:"
echo "    outputs/checkpoints/  — model .pt + history .pkl files"
echo "    outputs/logs/         — plots (.png), results (.pkl), reports (.txt)"
echo "════════════════════════════════════════════════════════════════════"
