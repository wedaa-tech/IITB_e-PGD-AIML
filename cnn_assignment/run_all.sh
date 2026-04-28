#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all.sh — end-to-end runner for the CIFAR-100 project.
#
# Runs all 8 entry-point scripts in the correct dependency order:
#
#   1  run_scratch.py                      (train scratch CNN)
#   2  run_transfer.py                     (train EfficientNet-B0 transfer)
#   3  compare.py                          (scratch vs transfer)
#   4  run_noise_robustness.py             (noise eval; writes noise_schedule.json)
#   5  run_noise_augment_training.py       (train noise-augmented scratch CNN)
#   6  compare_robustness_improvement.py   (baseline vs noise-augmented)
#   7  run_vgg_feature_mlp.py              (VGG16-BN features + MLP)
#   8  run_vgg_noise_robustness.py         (three-model noise comparison)
#
# Non-blocking plots:
#   By default the script sets MPLBACKEND=Agg, so plt.show() never opens a
#   window and never blocks execution — but every plt.savefig(...) still
#   writes PNGs to outputs/plots/. Use --interactive to opt into the old
#   blocking pop-up behaviour.
#
# Usage: ./run_all.sh [options]   (see --help)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")"

SKIP_PLOTS=0       # --no-plots:    skip plot generation entirely (fastest)
INTERACTIVE=0      # --interactive: pop up plot windows (BLOCKS until closed)
DRY_RUN=0          # --dry-run:     smoke-test each script
FROM_STEP=1
TO_STEP=8
ONLY_STEPS=""
LIST_ONLY=0

usage() {
cat <<'EOF'
Usage: ./run_all.sh [options]

By default:
  * Plots are generated and saved to outputs/plots/*.png
  * No GUI windows pop up — execution never blocks on a figure
  * PYTORCH_ENABLE_MPS_FALLBACK=1 is set (for Apple Silicon safety)

Options:
  --no-plots       Skip plot generation entirely (fastest; PNGs are NOT written).
                   Passes --no-plots to scripts that support it and still
                   keeps the non-interactive matplotlib backend for the rest.
  --interactive    Open plots in an interactive GUI window (BLOCKS until
                   you close each figure). Use only if you want to inspect
                   plots live as they are produced.
  --dry-run        Run each script with its built-in smoke-test flag
                   (few batches / 2 epochs per script — ~minutes total).
  --from N         Start at step N (1..8). Default: 1.
  --to   N         Stop after step N (1..8). Default: 8.
  --only "a b c"   Run only the listed step numbers, space-separated.
                   Example: --only "3 4 8"
  --list           Print the step list and exit.
  -h, --help       Show this help.

Steps:
  1  run_scratch.py                        train scratch CNN
  2  run_transfer.py                       train EfficientNet-B0
  3  compare.py                            scratch vs transfer
  4  run_noise_robustness.py               Gaussian noise eval (creates noise_schedule.json)
  5  run_noise_augment_training.py         noise-augmented training (scratch)
  6  compare_robustness_improvement.py     baseline vs noise-augmented
  7  run_vgg_feature_mlp.py                VGG16-BN features + MLP
  8  run_vgg_noise_robustness.py           three-model noise comparison

Examples:
  ./run_all.sh                             # full pipeline, plots saved, no blocking
  ./run_all.sh --no-plots                  # skip all plots entirely (fastest)
  ./run_all.sh --dry-run                   # quick smoke test of every step
  ./run_all.sh --only "3 4 8" --no-plots   # just comparisons, no plots
  ./run_all.sh --from 4                    # resume from noise robustness onwards
  ./run_all.sh --interactive               # old behaviour — plots pop up and block
EOF
}

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-plots)    SKIP_PLOTS=1; shift ;;
    --interactive) INTERACTIVE=1; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --from)        FROM_STEP="$2"; shift 2 ;;
    --to)          TO_STEP="$2"; shift 2 ;;
    --only)        ONLY_STEPS="$2"; shift 2 ;;
    --list)        LIST_ONLY=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

SCRIPTS=(
  "run_scratch.py"
  "run_transfer.py"
  "compare.py"
  "run_noise_robustness.py"
  "run_noise_augment_training.py"
  "compare_robustness_improvement.py"
  "run_vgg_feature_mlp.py"
  "run_vgg_noise_robustness.py"
)

LABELS=(
  "train scratch CNN"
  "train EfficientNet-B0 transfer"
  "compare scratch vs transfer"
  "Gaussian noise eval (creates noise_schedule.json)"
  "noise-augmented training (scratch)"
  "baseline vs noise-augmented"
  "VGG16-BN features + MLP"
  "three-model noise comparison"
)

if [[ "$LIST_ONLY" -eq 1 ]]; then
  for i in "${!SCRIPTS[@]}"; do
    printf "  %d  %-40s  %s\n" "$((i + 1))" "${SCRIPTS[$i]}" "${LABELS[$i]}"
  done
  exit 0
fi

# ── Environment setup ────────────────────────────────────────────────────────
export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}

if [[ "$INTERACTIVE" -ne 1 ]]; then
  # Non-interactive matplotlib backend: plt.show() becomes a no-op, but
  # every plt.savefig(...) still writes to disk. This is what prevents
  # scripts from blocking on plot windows.
  export MPLBACKEND=Agg
fi

# Auto-activate venv if we're not already inside one.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
  else
    echo "[run_all] No virtualenv active and ./venv not found. Run:" >&2
    echo "          python3 -m venv venv && source venv/bin/activate \\"  >&2
    echo "                                && pip install -r requirements.txt" >&2
    exit 1
  fi
fi

# ── Per-script flag assembly ─────────────────────────────────────────────────
# Scripts that accept --no-plots (the rest ignore plot generation because
# MPLBACKEND=Agg already suppresses windows, and skipping their savefig
# calls would need code changes we don't want to make here):
NO_PLOTS_CAPABLE=(
  "compare.py"
  "compare_robustness_improvement.py"
  "run_noise_robustness.py"
  "run_vgg_feature_mlp.py"
  "run_vgg_noise_robustness.py"
)

supports_no_plots () {
  local script="$1" s
  for s in "${NO_PLOTS_CAPABLE[@]}"; do
    [[ "$s" == "$script" ]] && return 0
  done
  return 1
}

build_flags () {
  # Build as a plain string (not an array). Expanding an empty array under
  # `set -u` trips "unbound variable" on macOS bash 3.2, which is what ships
  # with every recent macOS. The caller word-splits the result, so a string
  # is the right shape anyway.
  local script="$1"
  local flags=""
  if [[ "$SKIP_PLOTS" -eq 1 ]] && supports_no_plots "$script"; then
    flags+="--no-plots "
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    flags+="--dry-run "
  fi
  printf '%s' "$flags"
}

# ── Step runner ──────────────────────────────────────────────────────────────
should_run_step () {
  local n="$1"
  if [[ -n "$ONLY_STEPS" ]]; then
    for s in $ONLY_STEPS; do [[ "$s" == "$n" ]] && return 0; done
    return 1
  fi
  [[ "$n" -ge "$FROM_STEP" && "$n" -le "$TO_STEP" ]]
}

OVERALL_START=$(date +%s)
RAN_STEPS=()
SKIPPED_STEPS=()
FAILED_STEP=""

run_step () {
  local n="$1" script="$2" label="$3"

  if ! should_run_step "$n"; then
    SKIPPED_STEPS+=("$n")
    printf "\n[run_all] --- skipping step %d (%s) ---\n" "$n" "$script"
    return 0
  fi

  local flags
  flags="$(build_flags "$script")"

  echo
  echo "════════════════════════════════════════════════════════════════════"
  printf "  Step %d: %s — %s\n" "$n" "$script" "$label"
  printf "  Command: python %s %s\n" "$script" "$flags"
  echo "════════════════════════════════════════════════════════════════════"

  local t0 t1 dt
  t0=$(date +%s)
  if ! python "$script" $flags; then
    FAILED_STEP="$n ($script)"
    t1=$(date +%s); dt=$((t1 - t0))
    echo "[run_all] Step $n FAILED after ${dt}s — stopping pipeline." >&2
    return 1
  fi
  t1=$(date +%s); dt=$((t1 - t0))
  RAN_STEPS+=("$n")
  printf "[run_all] Step %d complete in %ds\n" "$n" "$dt"
}

cleanup_and_report () {
  local total=$(( $(date +%s) - OVERALL_START ))
  echo
  echo "════════════════════════════════════════════════════════════════════"
  echo "  Pipeline summary"
  echo "════════════════════════════════════════════════════════════════════"
  echo "  Total time : ${total}s"
  echo "  Executed   : ${RAN_STEPS[*]:-none}"
  echo "  Skipped    : ${SKIPPED_STEPS[*]:-none}"
  if [[ -n "$FAILED_STEP" ]]; then
    echo "  FAILED at  : $FAILED_STEP"
  fi
  echo
  echo "  Artifacts  : outputs/checkpoints/, outputs/plots/, outputs/results/"
}
trap cleanup_and_report EXIT

for i in "${!SCRIPTS[@]}"; do
  run_step "$((i + 1))" "${SCRIPTS[$i]}" "${LABELS[$i]}"
done
