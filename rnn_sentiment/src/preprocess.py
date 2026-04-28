"""
Full preprocessing pipeline for IMDB sentiment classification.

Flow:
  raw text
    → clean  (lowercase, strip HTML, remove punctuation)
    → tokenise
    → remove stop-words (keep negations)
    → lemmatise
    → build vocab (train set only)
    → encode to integer sequences
    → pad / truncate to MAX_SEQ_LEN
"""

import re
import json
import pickle
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PROCESSED, VOCAB_SIZE, MAX_SEQ_LEN,
    PAD_TOKEN, UNK_TOKEN, SPECIAL_TOKENS,
    NEGATIONS, TRAIN_SIZE, VAL_SIZE, RANDOM_SEED
)
from src.utils import set_seed, progress

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Singletons (loaded once) ──────────────────────────────────────────────
_lemmatizer   = WordNetLemmatizer()
_stop_words   = set(stopwords.words("english")) - NEGATIONS


# ─────────────────────────────────────────────────────────────────────────────
# 1. Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Lowercase → strip HTML → remove non-alpha → collapse whitespace.
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)          # punctuation / digits
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tokenisation + stop-word removal + lemmatisation
# ─────────────────────────────────────────────────────────────────────────────

def lemmatize_token(token: str) -> str:
    """Lemmatise: verb form first, then noun form."""
    return _lemmatizer.lemmatize(
        _lemmatizer.lemmatize(token, pos="v"), pos="n"
    )


def tokenize(text: str) -> list[str]:
    """
    word_tokenize → drop stop-words (keep negations) → lemmatise.
    Returns a list of string tokens.
    """
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stop_words]
    tokens = [lemmatize_token(t) for t in tokens]
    return tokens


def preprocess_text(text: str) -> list[str]:
    """End-to-end: raw text → cleaned token list."""
    return tokenize(clean_text(text))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    """
    word → index and index → word mappings.
    Built from training corpus only.
    """

    def __init__(self, max_size: int = VOCAB_SIZE):
        self.max_size  = max_size
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self._counter: Counter = Counter()

    # ── building ──────────────────────────────────────────────────────────

    def fit(self, tokenized_docs: list[list[str]]) -> "Vocabulary":
        """
        Count tokens across all training documents,
        keep the top max_size, assign indices.
        """
        log.info("Building vocabulary …")
        for doc in progress(tokenized_docs, desc="  counting tokens"):
            self._counter.update(doc)

        vocab_raw  = len(self._counter)
        most_common = self._counter.most_common(self.max_size)

        # index 0 = PAD, 1 = UNK, then sorted by frequency
        self.word2idx = {tok: 0 for tok in SPECIAL_TOKENS}
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.word2idx[tok] = i

        for rank, (word, _) in enumerate(most_common):
            idx = rank + len(SPECIAL_TOKENS)
            self.word2idx[word] = idx

        self.idx2word = {v: k for k, v in self.word2idx.items()}

        log.info(f"  Raw vocabulary : {vocab_raw:,} tokens")
        log.info(f"  Kept (top-N)   : {len(self.word2idx):,} tokens "
                 f"(+ {len(SPECIAL_TOKENS)} special)")
        return self

    # ── encoding ──────────────────────────────────────────────────────────

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.word2idx[UNK_TOKEN]
        return [self.word2idx.get(t, unk) for t in tokens]

    def __len__(self) -> int:
        return len(self.word2idx)

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"  Vocab saved → {path}")

    @staticmethod
    def load(path: Path) -> "Vocabulary":
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sequence padding / truncation
# ─────────────────────────────────────────────────────────────────────────────

def pad_sequence(
    seq: list[int],
    max_len: int  = MAX_SEQ_LEN,
    pad_idx: int  = 0,
    truncate: str = "post",
    padding: str  = "pre",
) -> np.ndarray:
    """
    Truncate long sequences (post = from right end),
    then pad short sequences (pre = left side, so final hidden state
    sees real tokens, not padding).
    """
    if len(seq) > max_len:
        seq = seq[:max_len] if truncate == "post" else seq[-max_len:]

    pad_len = max_len - len(seq)
    if padding == "pre":
        return np.array([pad_idx] * pad_len + seq, dtype=np.int64)
    return np.array(seq + [pad_idx] * pad_len, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full pipeline — download, preprocess, split, save
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(force: bool = False) -> dict:
    """
    Run the complete preprocessing pipeline and cache results to disk.

    Returns a dict with keys:
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        vocab
    """
    cache_file = DATA_PROCESSED / "dataset.pkl"
    vocab_file = DATA_PROCESSED / "vocab.pkl"

    if cache_file.exists() and vocab_file.exists() and not force:
        log.info("Loading cached preprocessed dataset …")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        data["vocab"] = Vocabulary.load(vocab_file)
        _print_stats(data)
        return data

    set_seed()

    # ── 5a. Load IMDB via HuggingFace datasets ────────────────────────────
    log.info("Downloading IMDB dataset …")
    raw = load_dataset("imdb")
    train_texts  = raw["train"]["text"]
    train_labels = raw["train"]["label"]
    test_texts   = raw["test"]["text"]
    test_labels  = raw["test"]["label"]

    # ── 5b. Preprocess all splits ─────────────────────────────────────────
    log.info("Preprocessing training reviews …")
    train_tokens = [preprocess_text(t) for t in
                    progress(train_texts, desc="  train")]

    log.info("Preprocessing test reviews …")
    test_tokens  = [preprocess_text(t) for t in
                    progress(test_texts, desc="  test")]

    # ── 5c. Build vocabulary on train only ────────────────────────────────
    vocab = Vocabulary(max_size=VOCAB_SIZE).fit(train_tokens)
    vocab.save(vocab_file)

    # ── 5d. Encode to integer sequences ───────────────────────────────────
    log.info("Encoding sequences …")
    train_encoded = [vocab.encode(t) for t in train_tokens]
    test_encoded  = [vocab.encode(t) for t in test_tokens]

    # ── 5e. Pad / truncate ────────────────────────────────────────────────
    log.info(f"Padding sequences to length {MAX_SEQ_LEN} …")
    X_all_train = np.array([pad_sequence(s) for s in
                             progress(train_encoded, desc="  pad train")])
    X_test      = np.array([pad_sequence(s) for s in
                             progress(test_encoded,  desc="  pad test")])
    y_all_train = np.array(train_labels, dtype=np.int64)
    y_test      = np.array(test_labels,  dtype=np.int64)

    # ── 5f. Train / val split ─────────────────────────────────────────────
    log.info("Splitting train → train + validation …")
    X_train, X_val, y_train, y_val = train_test_split(
        X_all_train, y_all_train,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_all_train,
    )

    # ── 5g. Report truncation stats ───────────────────────────────────────
    _report_truncation(train_tokens + test_tokens)

    # ── 5h. Cache to disk ─────────────────────────────────────────────────
    data = dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        vocab=vocab,
    )
    with open(cache_file, "wb") as f:
        pickle.dump({k: v for k, v in data.items() if k != "vocab"}, f)
    log.info(f"  Dataset cached → {cache_file}")

    _print_stats(data)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _report_truncation(all_tokens: list[list[str]]):
    lengths     = [len(t) for t in all_tokens]
    truncated   = sum(1 for l in lengths if l > MAX_SEQ_LEN)
    pct         = truncated / len(lengths) * 100
    percentiles = np.percentile(lengths, [50, 75, 90, 95, 99])
    log.info("Sequence length stats (after preprocessing):")
    log.info(f"  p50={percentiles[0]:.0f}  p75={percentiles[1]:.0f}  "
             f"p90={percentiles[2]:.0f}  p95={percentiles[3]:.0f}  "
             f"p99={percentiles[4]:.0f}")
    log.info(f"  Reviews truncated at {MAX_SEQ_LEN}: "
             f"{truncated:,} / {len(lengths):,}  ({pct:.1f}%)")


def _print_stats(data: dict):
    log.info("─" * 50)
    log.info("Dataset ready:")
    log.info(f"  X_train : {data['X_train'].shape}  "
             f"pos={data['y_train'].sum():,}")
    log.info(f"  X_val   : {data['X_val'].shape}    "
             f"pos={data['y_val'].sum():,}")
    log.info(f"  X_test  : {data['X_test'].shape}   "
             f"pos={data['y_test'].sum():,}")
    if "vocab" in data:
        log.info(f"  Vocab   : {len(data['vocab']):,} tokens")
    log.info("─" * 50)