# CIFAR-100: Scratch CNN vs Transfer Learning vs VGG Features — with Noise Robustness

End-to-end study on CIFAR-100 comparing three classifiers and measuring
how well each one survives additive Gaussian noise:

1. A custom 4-block CNN trained from scratch.
2. EfficientNet-B0 fine-tuned from ImageNet (two-phase: head-only → full).
3. Frozen VGG16-BN feature extractor + small MLP head.

On top of accuracy comparisons, the project runs:

- A Gaussian-noise robustness sweep (σ² = 0.05, the assignment spec) and the
  assignment's full σ ladder (0, 0.05, 0.1, 0.2, 0.3).
- A noise-augmented retraining experiment (Mixup + per-batch pixel-space
  Gaussian noise on a configurable fraction of samples) to improve robustness.
- A three-way comparison plot across all models at every noise level.

---

## Requirements

- macOS 12 (Monterey) or later
- Python 3.10 / 3.11 / 3.12 (project has been run on 3.14 as well, with
  the caveat below in the troubleshooting section)
- Apple Silicon (M1/M2/M3) recommended — uses MPS GPU acceleration
- Intel Mac / Linux also work — fall back to CPU automatically
- ~6 GB free disk space (dataset + VGG feature cache + checkpoints)
- ~8 GB RAM minimum (16 GB recommended for the transfer and VGG stages)

---

## Project structure

```
cifar100_project/
│
├── src/
│   ├── config.py                 # hyperparameters, paths, device selection
│   ├── dataset.py                # loaders, transforms, train/val/test splits
│   ├── train.py                  # shared training loop + early stopping
│   ├── evaluate.py               # metrics, inference timing, plotting helpers
│   ├── utils.py                  # history I/O, system info, misc plotting
│   ├── noise.py                  # NoiseConfig + Gaussian-noise injection
│   ├── noise_augment.py          # NoisyAugmentConfig + BatchNoiseAugmenter + Mixup
│   └── models/
│       ├── scratch_cnn.py        # 4-block custom CNN + variants
│       ├── transfer_model.py     # EfficientNet-B0 backbone + head utilities
│       └── vgg_extractor.py      # VGG16-BN feature extractor + MLP classifier
│
├── outputs/
│   ├── checkpoints/              # .pth model files (auto-created)
│   ├── plots/                    # .png figures (auto-created)
│   └── results/                  # .json / .csv logs (auto-created)
│
├── data/
│   ├── cifar-100-python/         # CIFAR-100 pickles (auto-downloaded)
│   ├── cifar-100-python.tar.gz   # original tarball (kept for re-extraction)
│   └── vgg_feature_cache/        # cached VGG features (auto-created in step 7)
│
├── run_scratch.py                # Step 1: train scratch CNN
├── run_transfer.py               # Step 2: train EfficientNet-B0 transfer
├── compare.py                    # Step 3: scratch vs transfer
├── run_noise_robustness.py       # Step 4: noise eval (writes noise_schedule.json)
├── run_noise_augment_training.py # Step 5: noise-augmented retraining
├── compare_robustness_improvement.py  # Step 6: baseline vs noise-augmented
├── run_vgg_feature_mlp.py        # Step 7: VGG features + MLP pipeline
├── run_vgg_noise_robustness.py   # Step 8: three-model noise comparison
│
├── run_all.sh                    # end-to-end runner with non-blocking plots
├── requirements.txt
└── README.md
```

---

## One-time setup

From the project root (`cifar100_project/`):

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Confirm the GPU is visible (Apple Silicon only):

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

The CIFAR-100 dataset is bundled under `data/cifar-100-python/`. If it goes
missing, either let `run_scratch.py` re-download it, or re-extract the
tarball manually:

```bash
cd data && tar -xzf cifar-100-python.tar.gz && cd ..
```

---

## Running everything at once (recommended)

Use the runner:

```bash
./run_all.sh
```

Default behaviour:

- Runs all 8 steps in dependency order (below).
- **Plots are generated and saved to `outputs/plots/*.png`**.
- **Plots never block execution** — the runner sets `MPLBACKEND=Agg`, which
  turns `plt.show()` into a no-op while still writing every `savefig(...)`
  to disk.
- Sets `PYTORCH_ENABLE_MPS_FALLBACK=1` for Apple Silicon safety.
- Auto-activates `./venv` if you haven't sourced it yet.

Common flags:

| Flag | Effect |
|---|---|
| `--no-plots` | Skip plot generation entirely (fastest). Passes `--no-plots` to scripts that support it. |
| `--interactive` | **Opt in** to pop-up windows — blocks execution until each figure is closed (old behaviour). |
| `--dry-run` | Pass every script's built-in smoke-test flag (few batches / 2 epochs). Good for validating the pipeline in ~minutes. |
| `--from N` | Start at step `N` (1..8). |
| `--to N` | Stop after step `N`. |
| `--only "3 4 8"` | Run only the listed step numbers. |
| `--list` | Print the step table and exit. |
| `-h / --help` | Usage. |

### Usage recipes

The runner is designed so you almost never need to type the 8 `python
run_*.py` commands by hand. Pick the recipe that matches your situation.

#### First-time full run (recommended default)

Trains every model from scratch, evaluates everything, and saves all plots
to `outputs/plots/` — without ever popping open a GUI window.

```bash
./run_all.sh
```

- Total time (MPS): ~3–5 hours. Training dominates (steps 1, 2, 5, 7).
- Safe to run in a detached terminal / `nohup` / `tmux`; nothing blocks.

#### Fastest possible run (skip plot generation entirely)

Passes `--no-plots` to every script that supports it, so the plotting code
paths are skipped as well (not just suppressed).

```bash
./run_all.sh --no-plots
```

- Fastest end-to-end path if you only care about JSON/CSV metrics.
- Note: steps 1, 2, and 5 don't yet expose a `--no-plots` flag, so they
  still generate their plots. `MPLBACKEND=Agg` ensures the PNGs are saved
  silently without blocking. Training time is what it is either way.

#### Smoke-test the whole pipeline in minutes

Each script has a `--dry-run` flag that uses only a couple of batches or
epochs. The runner passes it to every step.

```bash
./run_all.sh --dry-run
```

- Runs in ~5–15 minutes on MPS.
- Perfect after you've changed something and want to know the pipeline
  still wires up end-to-end before committing to a multi-hour training run.

Combine with `--no-plots` for the absolute minimum:

```bash
./run_all.sh --dry-run --no-plots
```

#### Old blocking behaviour (pop-up plot windows)

If you actually want to see plots appear one by one in a GUI window and
inspect them as they're produced:

```bash
./run_all.sh --interactive
```

- Each plot opens in a window and **blocks execution** until you close it.
- Handy for debugging a specific plot; not recommended for full runs.

#### Resume after a failure

If step 5 crashes, fix the issue and continue from there — no need to
re-train the earlier models:

```bash
./run_all.sh --from 5
```

You can also upper-bound the run:

```bash
./run_all.sh --from 3 --to 4        # just run compare + noise eval
```

#### Re-run just specific steps

Use `--only` with a space-separated list of step numbers. Useful once
checkpoints are in place and you're iterating on the comparison / plotting
logic.

```bash
./run_all.sh --only "3"                  # re-run compare.py only
./run_all.sh --only "3 6 8"              # all three comparison scripts
./run_all.sh --only "3 6 8" --no-plots   # comparisons, metrics only
./run_all.sh --only "4 8"                # refresh noise robustness reports
```

#### Training-only vs analysis-only passes

Split the pipeline cleanly — first train everything, then come back and do
all the analysis without re-training:

```bash
# Training only (steps 1, 2, 5, 7 — the ones that produce .pth files).
./run_all.sh --only "1 2 5 7"

# Analysis / comparison / plotting only (steps 3, 4, 6, 8).
./run_all.sh --only "3 4 6 8"
```

#### Headless / remote machines

`MPLBACKEND=Agg` is already set by default, so the runner works fine over
SSH without an X server or on CI. If you want to be explicit or run a
single script manually on a headless box:

```bash
MPLBACKEND=Agg PYTORCH_ENABLE_MPS_FALLBACK=1 python run_scratch.py
```

#### Inspect steps without running anything

Print the step table and exit (handy before composing an `--only` list):

```bash
./run_all.sh --list
./run_all.sh --help     # full usage
```

#### What the runner does for you automatically

- Sets `MPLBACKEND=Agg` so `plt.show()` never blocks (unless `--interactive`).
- Sets `PYTORCH_ENABLE_MPS_FALLBACK=1` for Apple Silicon safety.
- Sources `./venv/bin/activate` if you forgot to do it yourself.
- Prints a clear banner for each step with the exact command it's running.
- Times every step and prints a summary at the end — including which step
  failed, so resuming with `--from N` is trivial.

If any step fails, the runner prints which one and stops. The summary at
exit tells you exactly which step numbers succeeded, which were skipped,
and which one (if any) failed — so continuing with `./run_all.sh --from N`
(where `N` is the failed step) picks up right where you left off.

---

## Pipeline overview

| # | Script | Purpose | Produces | Requires |
|---|---|---|---|---|
| 1 | `run_scratch.py` | Train scratch CNN | `scratch_best.pth`, `scratch_standard_history.json` | CIFAR-100 data |
| 2 | `run_transfer.py` | Train EfficientNet-B0 transfer (2 phases) | `transfer_best.pth`, `transfer_*_history.json` | CIFAR-100 data |
| 3 | `compare.py` | Side-by-side eval of 1 vs 2 | `comparison_table.csv`, per-class CSVs, confusion matrices, robustness JSONs | 1 + 2 |
| 4 | `run_noise_robustness.py` | Assignment-spec Gaussian-noise eval of 1 & 2 | `noise_schedule.json`, `noise_results_{scratch,transfer}.json`, plots | 1 + 2 |
| 5 | `run_noise_augment_training.py` | Retrain scratch CNN with noise + Mixup augmentation | `scratch_noisy_augmented_best.pth`, `noise_augment_config_scratch.json` | CIFAR-100 data |
| 6 | `compare_robustness_improvement.py` | Baseline vs noise-augmented model | comparison JSONs + plots | 1 + 5 (+ 4 for the schedule) |
| 7 | `run_vgg_feature_mlp.py` | VGG16-BN feature extraction + MLP head | `vgg_mlp_best.pth`, cached features in `data/vgg_feature_cache/` | CIFAR-100 data |
| 8 | `run_vgg_noise_robustness.py` | Three-model noise comparison | `vgg_noise_results.json`, `full_robustness_report.json`, plots | 1 + 2 + 7 (+ 4) |

Each step is idempotent given its input checkpoints — you can re-run any
downstream step without retraining earlier ones as long as the required
`.pth` and JSON artefacts are still on disk.

---

## Running steps manually

Every script is a standalone entry point. Run from the project root with
the venv activated. Pass `--help` to any of them for the full flag list.

### Step 1 — Train the scratch CNN

```bash
python run_scratch.py
```

- Splits CIFAR-100 training set into 45,000 train / 5,000 validation.
- Trains a 4-block CNN for up to 50 epochs with early stopping.
- Saves `outputs/checkpoints/scratch_best.pth`.
- Runtime: ~45–70 min on MPS, 3–5 h on CPU.

### Step 2 — Train the EfficientNet-B0 transfer model

```bash
python run_transfer.py
```

- Phase A (15 epochs): freezes the backbone, trains only the classifier head at lr=1e-3.
- Phase B (25 epochs): unfreezes everything, discriminative lrs (backbone 1e-5, head 1e-3).
- Saves `outputs/checkpoints/transfer_best.pth`.
- Runtime: ~60–90 min on MPS, 4–7 h on CPU.

### Step 3 — Compare scratch vs transfer

```bash
python compare.py                 # full evaluation
python compare.py --quick         # 1-batch sanity run
python compare.py --no-plots      # metrics only
```

- Top-1 / top-5 accuracy, per-class breakdown, confusion matrices,
  Gaussian-noise robustness at σ ∈ {0, 0.05, 0.1, 0.2, 0.3}, inference timing.
- Writes `outputs/results/comparison_table.csv` and several plots.

### Step 4 — Assignment-spec noise robustness

```bash
python run_noise_robustness.py
python run_noise_robustness.py --dry-run
python run_noise_robustness.py --save-noisy-images --verify-noise
```

- Applies zero-mean Gaussian noise with σ² = 0.05 in [0,1] pixel space, clipped.
- **Writes `outputs/results/noise_schedule.json`** — reused by steps 6 and 8.

### Step 5 — Noise-augmented training

```bash
python run_noise_augment_training.py                  # scratch CNN (default)
python run_noise_augment_training.py --model transfer # transfer variant
python run_noise_augment_training.py --noise-prob 0.3 --no-mixup
```

- Per-batch mixed training: a fraction of samples receive Gaussian noise
  (σ² = 0.05 by default, i.e. same as the robustness spec); optional Mixup.
- Saves `outputs/checkpoints/scratch_noisy_augmented_best.pth` (kept separate
  from the baseline `scratch_best.pth` so you can compare them).

### Step 6 — Baseline vs noise-augmented

```bash
python compare_robustness_improvement.py
python compare_robustness_improvement.py --model transfer
```

- Evaluates both checkpoints on clean and noisy test sets.
- Reports accuracy retention (noisy/clean ratio), which is the fair metric
  for comparing robustness across models with different clean accuracies.

### Step 7 — VGG16-BN features + MLP

```bash
python run_vgg_feature_mlp.py                    # full pipeline
python run_vgg_feature_mlp.py --skip-extraction  # reuse cached features
python run_vgg_feature_mlp.py --vgg vgg11_bn     # lighter backbone
```

- Extracts frozen VGG16-BN features for train/val/test and caches them to
  `data/vgg_feature_cache/` on first run (`--skip-extraction` reuses the cache).
- Trains an MLP head on the cached features.
- Saves `outputs/checkpoints/vgg_mlp_best.pth`.

### Step 8 — Three-model noise comparison

```bash
python run_vgg_noise_robustness.py
python run_vgg_noise_robustness.py --sigma-levels 0.0 0.05 0.1 0.2 0.3
```

- Injects noise at pixel level **before** VGG feature extraction (so this
  measures true image-space robustness, not feature-space robustness).
- Aggregates results for scratch / transfer / VGG+MLP into
  `outputs/results/full_robustness_report.json` plus comparison plots.

---

## Outputs

After the full pipeline, `outputs/` looks like:

```
outputs/
├── checkpoints/
│   ├── scratch_best.pth                    # step 1
│   ├── transfer_best.pth                   # step 2
│   ├── transfer_phase_a_best.pth           # step 2 (phase-A snapshot)
│   ├── scratch_noisy_augmented_best.pth    # step 5
│   └── vgg_mlp_best.pth                    # step 7
│
├── plots/
│   ├── learning_curves.png                 # step 3
│   ├── confusion_{scratch,transfer}.png    # step 3
│   ├── robustness.png                      # step 3
│   ├── noise_sample_images.png             # step 4
│   ├── noise_accuracy_comparison.png       # step 4
│   ├── noise_confidence_shift.png          # step 4
│   ├── full_robustness_comparison.png      # step 8
│   ├── robustness_retention_chart.png      # step 8
│   └── per_model_accuracy_drop.png         # step 8
│
└── results/
    ├── scratch_standard_history.json       # step 1
    ├── transfer_efficientnet_b0_history.json  # step 2
    ├── comparison_table.csv                # step 3
    ├── per_class_{scratch,transfer}.csv    # step 3
    ├── robustness_{scratch,transfer}.json  # step 3
    ├── noise_schedule.json                 # step 4 (consumed by steps 6 & 8)
    ├── noise_results_{scratch,transfer}.json  # step 4
    ├── scratch_noisy_augmented_history.json   # step 5
    ├── noise_augment_config_scratch.json   # step 5
    ├── vgg_mlp_history.json                # step 7
    ├── vgg_mlp_results.json                # step 7
    ├── vgg_noise_results.json              # step 8
    └── full_robustness_report.json         # step 8
```

---

## Expected headline results

| Metric (CIFAR-100 test set, 10,000 images) | Scratch CNN | EfficientNet-B0 (transfer) | VGG16-BN + MLP |
|---|---|---|---|
| Top-1 accuracy | ~55–62% | ~73–80% | ~65–72% |
| Top-5 accuracy | ~80–85% | ~92–95% | ~88–92% |
| Parameters (trainable) | ~9.2M | ~5.3M | ~0.4M (MLP head only) |
| Robustness at σ=0.1 (retention) | lowest | highest | middle |

The transfer model wins on both clean accuracy and robustness because
ImageNet pretraining provides strong low-level features (edges, textures)
that generalise well to CIFAR-100 after upscaling. The VGG+MLP pipeline
sits in between: strong features, but the small trainable head limits
upside. The scratch CNN has the largest number of trainable parameters
but the weakest inductive priors.

---

## Troubleshooting

### Plots block the terminal

`plt.show()` opens a GUI window and waits for you to close it. Two fixes:

- **Recommended:** use `./run_all.sh` — it sets `MPLBACKEND=Agg` so
  `plt.show()` becomes a no-op but `savefig(...)` still writes PNGs.
- **One-off:** run any script with the same variable inline:

  ```bash
  MPLBACKEND=Agg python run_scratch.py
  ```

### MPS errors on Apple Silicon

Some PyTorch ops fall back from MPS gracefully; a few raise
`NotImplementedError`. Enable the CPU fallback:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python <script>.py
```

The runner already sets this.

### `Expected a 'mps' device type for generator but found 'cpu'`

Noise-injection code (`src/noise.py`, `src/noise_augment.py`,
`run_vgg_noise_robustness.py`) samples noise on CPU with a seeded
`torch.Generator` and moves it to the active device. If you add new
noise code elsewhere, follow the same pattern — MPS rejects CPU
generators in `tensor.normal_(..., generator=...)`.

### Multiprocessing warning on Mac

`num_workers=0` is already hard-coded in `src/config.py` to avoid the
`RuntimeError: An attempt has been made to start a new process` that
appears on macOS spawn-based workers. Leave it at 0 unless you know your
environment handles it.

### Running out of RAM during transfer learning

Lower the transfer batch size in `src/config.py`:

```python
TRANSFER = {
    ...
    "batch_size": 32,   # reduce from 64
    ...
}
```

### Dataset re-download

```bash
rm -rf data/cifar-100-python
python run_scratch.py        # auto re-downloads
# OR re-extract from the bundled tarball:
cd data && tar -xzf cifar-100-python.tar.gz && cd ..
```

### Python 3.14

The venv was originally created on Python 3.14. Some C-extension packages
(matplotlib, sklearn, torchvision) don't publish 3.14 wheels yet — if
installation fails, create the venv on Python 3.11 instead:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Running a script directly as `python src/dataset.py`

Some `src/*.py` files live inside the `src` package and use absolute
imports like `from src.config import ...`. Prefer running them as modules:

```bash
python -m src.dataset
```

Running `python src/dataset.py` adds `src/` (not the project root) to
`sys.path`, which breaks the absolute imports. `src/dataset.py` includes
a small bootstrap to handle both styles, but `-m` is the cleanest path.

---

## Deactivating / reactivating the environment

```bash
deactivate                       # leave the venv
source venv/bin/activate         # re-enter it
```

---

## References

- CIFAR-100: Krizhevsky, A. (2009). *Learning Multiple Layers of Features
  from Tiny Images*.
- EfficientNet: Tan & Le (2019). *EfficientNet: Rethinking Model Scaling
  for Convolutional Neural Networks*. ICML 2019.
- VGG: Simonyan & Zisserman (2015). *Very Deep Convolutional Networks for
  Large-Scale Image Recognition*. ICLR 2015.
- Mixup: Zhang et al. (2018). *mixup: Beyond Empirical Risk Minimization*.
  ICLR 2018.
- PyTorch documentation: <https://pytorch.org/docs>
- timm model library: <https://huggingface.co/docs/timm>
