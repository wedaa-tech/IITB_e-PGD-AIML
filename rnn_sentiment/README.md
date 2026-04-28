# IMDB Sentiment: Vanilla RNN vs LSTM vs Attention-LSTM — with Embedding Ablations

End-to-end study on the IMDB Large Movie Review Dataset (Maas et al., 2011)
comparing three recurrent architectures and three token-vectorisation
strategies for binary sentiment classification:

1. A vanilla `RNN` classifier (B3).
2. An `LSTM` classifier with the same `(L, h)` shape (B5).
3. An attention-augmented `AttentionLSTM` (B6).

Each architecture is trained under three different embedding strategies —
a **learned 128-dim embedding**, **pretrained Word2Vec** (trained on the IMDB
training corpus), and **pretrained GloVe-100d** (Common Crawl) — and a full
`L × h` hyperparameter sweep is run for the RNN and LSTM variants.

On top of architecture and embedding comparisons, the project produces:

- A 3 × 2 grid of `{RNN, LSTM, Attention} × {Word2Vec, GloVe}` runs (B4).
- A `(L, h) ∈ {1, 2, 3} × {128, 256, 512}` sweep with per-epoch gradient-norm
  tracking — useful for visualising vanishing gradients in the deep RNN.
- Attention-weight heatmaps over real IMDB reviews and an attention entropy
  / overhead study (B6).

---

## Requirements

- macOS 12 (Monterey) or later
- Python 3.10 / 3.11 / 3.12 (project has been run on 3.14 as well, with
  the caveat below in the troubleshooting section)
- Apple Silicon (M1/M2/M3) recommended — uses MPS GPU acceleration
- Intel Mac / Linux also work — fall back to CPU automatically
- ~3 GB free disk space (IMDB cache + GloVe vectors + checkpoints)
- ~8 GB RAM minimum (16 GB recommended for the full hparam sweep)

---

## Project structure

```
rnn_sentiment/
│
├── src/
│   ├── config.py                 # hyperparameters, paths, device selection
│   ├── preprocess.py             # cleaning, tokenisation, vocab, padding
│   ├── dataset.py                # PyTorch Datasets & DataLoaders
│   ├── embeddings.py             # one-hot, Word2Vec, GloVe loaders
│   ├── trainer.py                # shared training loop + grad-norm tracking
│   ├── metrics.py                # accuracy / loss / classification metrics
│   ├── hparam_search.py          # L x h sweep over RNN + LSTM
│   ├── visualize.py              # plotting helpers (curves, confusion, bars)
│   ├── utils.py                  # seed, device info, progress helpers
│   └── models/
│       ├── rnn.py                # vanilla RNN classifier
│       ├── lstm.py               # LSTM classifier
│       └── attention_lstm.py     # LSTM + additive attention
│
├── outputs/
│   ├── checkpoints/              # .pt + .pkl history files (auto-created)
│   └── logs/                     # plots (.png), results (.pkl), reports (.txt)
│
├── data/
│   ├── raw/                      # HuggingFace IMDB cache (auto-downloaded)
│   ├── processed/                # tokenised splits, vocab, embedding caches
│   │   ├── dataset.pkl
│   │   ├── vocab.pkl
│   │   ├── tokenized_train.pkl
│   │   ├── word2vec_matrix.npy
│   │   └── glove_matrix.npy
│   └── glove/                    # pretrained GloVe vectors (manual download)
│       └── glove.6B.100d.txt
│
├── main.py                       # Step 1: train RNN / LSTM / Attention (learned emb)
├── experiment.py                 # Step 2: 3 models x {Word2Vec, GloVe} grid
├── run_hparam.py                 # Step 3: L x h sweep (RNN + LSTM)
├── analyze.py                    # Step 4: baseline plots + reports
├── analyze_b4.py                 # Step 5: B4 embedding-comparison plots
├── analyze_b5.py                 # Step 6: B5 RNN vs LSTM plots
├── analyze_b6.py                 # Step 7: B6 attention plots
│
├── run_all.sh                    # end-to-end runner with non-blocking plots
├── requirements.txt
└── README.md
```

---

## One-time setup

From the project root (`rnn_sentiment/`):

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

Download the NLTK resources used by `src/preprocess.py` (stopwords,
WordNet lemmatiser, Punkt tokeniser) — only needed once:

```bash
python -c "import nltk; [nltk.download(p) for p in ('stopwords','wordnet','omw-1.4','punkt','punkt_tab')]"
```

The IMDB dataset is fetched lazily via the HuggingFace `datasets` library
the first time `run_pipeline()` runs and is cached under
`data/raw/`. Subsequent runs reuse the processed pickles in
`data/processed/`.

GloVe vectors are **not** bundled (the file is ~330 MB). Download
`glove.6B.100d.txt` once and drop it in `data/glove/`:

```bash
mkdir -p data/glove
cd data/glove
curl -O https://nlp.stanford.edu/data/glove.6B.zip
unzip -o glove.6B.zip "glove.6B.100d.txt"
cd ../..
```

Word2Vec is trained on the IMDB training corpus on first use and cached
to `data/processed/word2vec_matrix.npy`, so it has no external download.

---

## Running everything at once (recommended)

Use the runner:

```bash
./run_all.sh
```

Default behaviour:

- Runs all 7 steps in dependency order (below).
- **Plots are generated and saved to `outputs/logs/*.png`**.
- **Plots never block execution** — the runner sets `MPLBACKEND=Agg`, which
  turns `plt.show()` into a no-op while still writing every `savefig(...)`
  to disk.
- Sets `PYTORCH_ENABLE_MPS_FALLBACK=1` for Apple Silicon safety.
- Auto-activates a `conda` env named `rnn_sentiment` if it exists, else
  falls back to `./venv` if you haven't sourced it yet.

Common flags:

| Flag | Effect |
|---|---|
| `-e N`, `--epochs N` | Train every model for `N` epochs (sets `RNN_EPOCHS=N`). The default is 10 (see `src/config.py`). Propagated to `main.py`, `experiment.py`, and `src/hparam_search.py`. |
| `--from K` | Start at step `K` (1..7). |
| `--to K` | Stop after step `K`. |
| `--only LIST` | Comma-separated list of step numbers (e.g. `1,3,5`). |
| `--list` | Print the step table and exit. |
| `--dry-run` | Print the plan without executing anything. |
| `-h / --help` | Usage. |

### Usage recipes

The runner is designed so you almost never need to type the 7 `python
*.py` commands by hand. Pick the recipe that matches your situation.

#### First-time full run (recommended default)

Trains every model from scratch, evaluates everything, and saves all plots
to `outputs/logs/` — without ever popping open a GUI window.

```bash
./run_all.sh
```

- Total time (MPS, 10 epochs): ~3–4 hours. The hparam sweep (step 3)
  dominates because it trains 18 configurations (9 RNN + 9 LSTM).
- Safe to run in a detached terminal / `nohup` / `tmux`; nothing blocks.

#### Faster smoke run (fewer epochs)

Cut training time roughly in half by lowering the epoch count for every
training step:

```bash
./run_all.sh --epochs 5
./run_all.sh -e 3              # really fast end-to-end shake-out
```

The analysis steps (4–7) ignore `--epochs` since they don't train.

#### Inspect the plan without running anything

```bash
./run_all.sh --list
./run_all.sh --dry-run
./run_all.sh --help
```

`--list` prints the 7-step table and exits. `--dry-run` prints the exact
command for each step but never executes them — handy for confirming a
custom `--from`/`--to`/`--only` selection before committing.

#### Resume after a failure

If step 3 (the hparam sweep) crashes part-way, fix the issue and continue
from there — no need to re-train the baseline models from step 1:

```bash
./run_all.sh --from 3
```

You can also upper-bound the run:

```bash
./run_all.sh --from 1 --to 2 -e 5    # only baseline + embedding grid, 5 epochs
```

#### Re-run just specific steps

Use `--only` with a comma-separated list of step numbers. Useful once
checkpoints are in place and you're iterating on a specific analysis script.

```bash
./run_all.sh --only 4                  # re-run analyze.py only
./run_all.sh --only 5,6,7              # all three B4/B5/B6 analyses
./run_all.sh --only 1,2 -e 8           # re-train baseline + embedding grid for 8 epochs
./run_all.sh --only 7                  # refresh attention plots after editing analyze_b6.py
```

#### Training-only vs analysis-only passes

Split the pipeline cleanly — first train everything, then come back and do
all the analysis without re-training:

```bash
# Training only (steps 1, 2, 3 — the ones that produce .pt / .pkl files).
./run_all.sh --only 1,2,3

# Analysis / plotting only (steps 4, 5, 6, 7).
./run_all.sh --only 4,5,6,7
```

#### Headless / remote machines

`MPLBACKEND=Agg` is already set by default, so the runner works fine over
SSH without an X server or on CI. If you want to be explicit or run a
single script manually on a headless box:

```bash
MPLBACKEND=Agg PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py
```

#### What the runner does for you automatically

- Sets `MPLBACKEND=Agg` so `plt.show()` never blocks.
- Sets `PYTORCH_ENABLE_MPS_FALLBACK=1` for Apple Silicon safety.
- Activates a `conda` env named `rnn_sentiment` if it exists, else
  sources `./venv/bin/activate` if present, else falls back to whatever
  `python` is on `PATH`.
- Exports `RNN_EPOCHS` so a single `--epochs N` flag is honoured by all
  three training entry points (`main.py`, `experiment.py`,
  `src/hparam_search.py`).
- Prints a clear banner for each step with the exact command it's running.
- Times every step and prints a summary at the end — including which step
  failed, so resuming with `--from K` is trivial.

If any step fails, the runner prints which one and stops with a non-zero
exit code. The summary at exit tells you exactly which step numbers
succeeded, which were skipped, and which one (if any) failed — so
continuing with `./run_all.sh --from K` (where `K` is the failed step)
picks up right where you left off.

---

## Pipeline overview

| # | Script | Purpose | Produces | Requires |
|---|---|---|---|---|
| 1 | `main.py` | Train baseline RNN / LSTM / Attention with a learned 128-dim embedding (B3, B5, B6 baseline) | `{rnn,lstm,attention}_best.pt`, `*_history.pkl` | IMDB data |
| 2 | `experiment.py` | `{RNN, LSTM, Attention} × {Word2Vec, GloVe}` grid (B4 + B5 pretrained-embedding comparison) | `{model}_{embedding}_best.pt`, `experiment_results.pkl` | IMDB data, GloVe |
| 3 | `run_hparam.py` | `L × h` sweep over RNN + LSTM (B3 hyperparameter study) | `{rnn,lstm}_L{L}_H{h}_best.pt`, `hparam_results.pkl` | IMDB data, GloVe |
| 4 | `analyze.py` | Baseline plots + classification reports for step 1 checkpoints | `training_curves.png`, `confusion_matrices.png`, `comparison_bar.png`, `classification_reports.txt` | 1 |
| 5 | `analyze_b4.py` | B4 embedding-comparison plots | `embedding_comparison.png`, B4 training curves | 2 |
| 6 | `analyze_b5.py` | B5 RNN-vs-LSTM plots (stability, convergence speed, full comparison) | `b5_rnn_vs_lstm_curves.png`, `b5_stability.png`, `b5_convergence_speed.png`, `b5_full_comparison.png` | 1, 2, 3 |
| 7 | `analyze_b6.py` | B6 attention study (heatmaps, overhead, entropy) | `b6_three_model_comparison.png`, `b6_attention_heatmaps.png`, `b6_attention_entropy.png`, `b6_all_curves.png` | 1, 2 |

Each step is idempotent given its input checkpoints — you can re-run any
downstream analysis step without retraining earlier ones as long as the
required `.pt` and `.pkl` artefacts are still on disk.

---

## Running steps manually

Every script is a standalone entry point. Run from the project root with
the venv activated. The training scripts honour the `RNN_EPOCHS` env var
(see `src/config.py`); the analysis scripts ignore it.

### Step 1 — Train the baseline models (learned embedding)

```bash
python main.py
RNN_EPOCHS=5 python main.py    # override default (10) epochs
```

- Trains `rnn`, `lstm`, and `attention` (each with `L=2`, `h=256`,
  `embed_dim=128`, `dropout=0.5`) on a 20k / 5k / 25k IMDB train / val /
  test split.
- Saves `outputs/checkpoints/{model}_best.pt` and `*_history.pkl`.
- Runtime: ~25–40 min on MPS, 2–3 h on CPU.

### Step 2 — Embedding ablation grid (B4)

```bash
python experiment.py
```

- Trains `{rnn, lstm, attention} × {word2vec, glove}` — 6 runs total —
  using the same `L=2`, `h=256` shape as step 1, and 100-dim pretrained
  vectors (so the embedding layer is non-learnable / partially fine-tuned).
- Word2Vec is trained on the IMDB train corpus on first use and cached.
- Saves per-config `*_best.pt` and a combined `experiment_results.pkl`.
- Runtime: ~50–80 min on MPS.

### Step 3 — Hyperparameter sweep (B3)

```bash
python run_hparam.py
```

- Sweeps `L ∈ {1, 2, 3} × h ∈ {128, 256, 512}` for both `rnn` and `lstm`
  (18 configurations) on top of the GloVe embedding.
- Tracks per-layer gradient norms each epoch, so the resulting plots make
  the vanishing-gradient problem in the deeper RNN visible directly.
- Caches everything to `outputs/logs/hparam_results.pkl`; rerunning skips
  configs whose checkpoint already exists. Delete the cache to force a
  fresh sweep.
- Runtime: ~90–150 min on MPS — this is the longest step.

### Step 4 — Baseline analysis

```bash
python analyze.py
```

- Loads the three baseline checkpoints from step 1, plots training
  curves, per-model confusion matrices, a comparison bar chart, and
  writes a per-class classification report to
  `outputs/logs/classification_reports.txt`.

### Step 5 — B4 analysis (embedding comparison)

```bash
python analyze_b4.py
```

- Reads `experiment_results.pkl` from step 2 and produces the
  embedding-comparison figure plus per-config training curves used in
  the B4 write-up.

### Step 6 — B5 analysis (RNN vs LSTM)

```bash
python analyze_b5.py
```

- Cross-references step 1, 2, and 3 results to compare RNN vs LSTM in
  terms of training stability, convergence speed, accuracy, and
  generalisation. Produces the four `b5_*.png` figures.

### Step 7 — B6 analysis (attention study)

```bash
python analyze_b6.py
```

- Loads the attention checkpoint, runs it over a handful of real IMDB
  reviews, and plots:
  - the three-model performance comparison,
  - per-token attention heatmaps,
  - an attention-entropy histogram,
  - the combined training curves figure.
- Also prints a parameter / epoch-time / memory overhead table.

---

## Outputs

After the full pipeline, `outputs/` looks like:

```
outputs/
├── checkpoints/
│   ├── rnn_best.pt                         # step 1
│   ├── lstm_best.pt                        # step 1
│   ├── attention_best.pt                   # step 1
│   ├── {rnn,lstm,attention}_history.pkl    # step 1
│   │
│   ├── {rnn,lstm,attention}_word2vec_best.pt   # step 2
│   ├── {rnn,lstm,attention}_glove_best.pt      # step 2
│   ├── {rnn,lstm,attention}_{w2v,glove}_history.pkl  # step 2
│   │
│   ├── {rnn,lstm}_L{1,2,3}_H{128,256,512}_best.pt    # step 3
│   └── {rnn,lstm}_L{1,2,3}_H{128,256,512}_history.pkl
│
└── logs/
    ├── training_curves.png                 # step 4
    ├── rnn_curve.png                       # step 4
    ├── lstm_curve.png                      # step 4
    ├── attention_curve.png                 # step 4
    ├── confusion_matrices.png              # step 4
    ├── confusion_matrices_summary.json     # step 4
    ├── comparison_bar.png                  # step 4
    ├── classification_reports.txt          # step 4
    │
    ├── embedding_comparison.png            # step 5
    ├── experiment_results.pkl              # step 2 (consumed by 5 & 6)
    │
    ├── b5_rnn_vs_lstm_curves.png           # step 6
    ├── b5_stability.png                    # step 6
    ├── b5_convergence_speed.png            # step 6
    ├── b5_full_comparison.png              # step 6
    │
    ├── b6_three_model_comparison.png       # step 7
    ├── b6_attention_heatmaps.png           # step 7
    ├── b6_attention_entropy.png            # step 7
    ├── b6_all_curves.png                   # step 7
    ├── b_partB_confusion_grid.png          # step 7
    │
    ├── hparam_heatmap.png                  # step 3
    ├── best_config_curves.png              # step 3
    ├── gradient_norms.png                  # step 3
    └── hparam_results.pkl                  # step 3
```

---

## Expected headline results

| Metric (IMDB test set, 25,000 reviews) | Vanilla RNN | LSTM | Attention-LSTM |
|---|---|---|---|
| Test accuracy (learned 128-dim embed) | ~70–78% | ~85–88% | ~86–89% |
| Test accuracy (GloVe 100d) | ~75–80% | ~87–89% | ~88–90% |
| Gradient norm at deepest layer (L=3) | very small (vanishing) | stable | stable |
| Trainable parameters (L=2, h=256, GloVe) | ~0.4M + emb | ~1.6M + emb | ~1.7M + emb |
| Convergence (epochs to best val acc) | slow / unstable | fast | fast |

The LSTM beats the vanilla RNN on every axis because the gating mechanism
preserves gradient flow through long reviews (≤500 tokens). The attention
mechanism gives a further consistent boost — small, but reliable — by
letting the classifier focus on a handful of opinion-bearing tokens rather
than relying on the final hidden state alone. GloVe ≥ Word2Vec ≥ learned
on the same architecture once you control for parameter count.

---

## Troubleshooting

### Plots block the terminal

`plt.show()` opens a GUI window and waits for you to close it. Two fixes:

- **Recommended:** use `./run_all.sh` — it sets `MPLBACKEND=Agg` so
  `plt.show()` becomes a no-op but `savefig(...)` still writes PNGs.
- **One-off:** run any script with the same variable inline:

  ```bash
  MPLBACKEND=Agg python main.py
  ```

### MPS errors on Apple Silicon

Some PyTorch ops fall back from MPS gracefully; a few raise
`NotImplementedError`. Enable the CPU fallback:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python <script>.py
```

The runner already sets this.

### NLTK `LookupError` (`Resource stopwords / wordnet not found`)

NLTK doesn't ship corpora — they're downloaded lazily into
`~/nltk_data/`. The first run in a fresh environment fails until you
populate them:

```bash
python -c "import nltk; [nltk.download(p) for p in ('stopwords','wordnet','omw-1.4','punkt','punkt_tab')]"
```

### Re-run preprocessing from scratch

The processed splits are cached under `data/processed/`. To force a clean
preprocessing pass (e.g. after changing `MAX_SEQ_LEN` or the vocab size):

```bash
rm -rf data/processed
python main.py        # re-tokenises and re-builds vocab
```

This also invalidates the Word2Vec / GloVe matrix caches, so they will
be rebuilt on the next embedding-aware run.

### GloVe file missing (`FileNotFoundError: glove.6B.100d.txt`)

`src/embeddings.py` expects `data/glove/glove.6B.100d.txt`. Download it
once:

```bash
mkdir -p data/glove
cd data/glove
curl -O https://nlp.stanford.edu/data/glove.6B.zip
unzip -o glove.6B.zip "glove.6B.100d.txt"
cd ../..
```

### Out of RAM during the hparam sweep

The `(L=3, h=512)` configurations are the largest. Reduce the batch size
in `src/config.py`:

```python
BATCH_SIZE = 32   # reduce from 64
```

…or restrict the grid in `src/hparam_search.py`:

```python
L_VALUES = [1, 2]            # drop L=3
H_VALUES = [128, 256]        # drop h=512
```

### Multiprocessing warning on Mac

The DataLoaders use `num_workers=0` by default to avoid the
`RuntimeError: An attempt has been made to start a new process` that
appears on macOS spawn-based workers. Leave it at 0 unless you know your
environment handles it.

### Python 3.14

Some C-extension packages (matplotlib, scikit-learn, gensim) don't
publish 3.14 wheels yet — if installation fails, create the venv on
Python 3.11 instead:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Running a script directly as `python src/preprocess.py`

`src/*.py` files use absolute imports like `from src.config import ...`.
Prefer running them as modules:

```bash
python -m src.preprocess
```

Running `python src/preprocess.py` adds `src/` (not the project root) to
`sys.path`, which breaks the absolute imports.

---

## Deactivating / reactivating the environment

```bash
deactivate                       # leave the venv
source venv/bin/activate         # re-enter it
```

---

## References

- IMDB dataset: Maas, A., et al. (2011). *Learning Word Vectors for
  Sentiment Analysis*. ACL 2011.
  <https://ai.stanford.edu/~amaas/data/sentiment/>
- Vanilla RNN: Elman, J. L. (1990). *Finding Structure in Time*.
  Cognitive Science, 14(2).
- LSTM: Hochreiter & Schmidhuber (1997). *Long Short-Term Memory*.
  Neural Computation, 9(8).
- Attention: Bahdanau, Cho & Bengio (2015). *Neural Machine Translation
  by Jointly Learning to Align and Translate*. ICLR 2015.
- Word2Vec: Mikolov et al. (2013). *Distributed Representations of Words
  and Phrases and their Compositionality*. NeurIPS 2013.
- GloVe: Pennington, Socher & Manning (2014). *GloVe: Global Vectors for
  Word Representation*. EMNLP 2014.
- PyTorch documentation: <https://pytorch.org/docs>
- HuggingFace `datasets`: <https://huggingface.co/docs/datasets>
