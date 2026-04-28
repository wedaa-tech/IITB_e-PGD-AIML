"""
Three token-to-vector strategies for the IMDB sentiment task.

Strategy        Dim         Trained on          Learnable?
──────────────────────────────────────────────────────────
One-Hot         20,002      N/A (identity)      No
Word2Vec        100         IMDB train corpus   Optional
GloVe           100         840B Common Crawl   Optional (fine-tune)
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config      import (
    BASE_DIR, DATA_PROCESSED, VOCAB_SIZE,
    PAD_TOKEN, UNK_TOKEN,
)
from src.preprocess  import Vocabulary
from src.utils       import progress

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
GLOVE_PATH = BASE_DIR / "data" / "glove" / "glove.6B.100d.txt"
W2V_CACHE  = DATA_PROCESSED / "word2vec_matrix.npy"
GLOVE_CACHE = DATA_PROCESSED / "glove_matrix.npy"

# ── Shared constants ──────────────────────────────────────────────────────────
REAL_VOCAB_SIZE = VOCAB_SIZE + 2    # +2 for <PAD> and <UNK>
W2V_DIM         = 100
GLOVE_DIM       = 100


# ─────────────────────────────────────────────────────────────────────────────
# 1. One-Hot Embedding
# ─────────────────────────────────────────────────────────────────────────────

def get_onehot_embedding(vocab_size: int = REAL_VOCAB_SIZE) -> nn.Embedding:
    """
    One-hot encoding via a fixed (non-trainable) identity-like embedding.

    Each token index i maps to a sparse binary vector of length vocab_size
    where position i = 1 and all others = 0.

    Embedding matrix : I  ∈ R^(vocab_size × vocab_size)
    Dimension        : 20,002
    Trainable        : No — frozen identity matrix.

    Limitation: No semantic similarity between tokens. Cosine similarity
    between any two distinct one-hot vectors is always 0. Memory-intensive
    at 20,002-dim inputs to the RNN.
    """
    weight = torch.eye(vocab_size)          # identity matrix
    emb    = nn.Embedding(vocab_size, vocab_size, padding_idx=0)
    emb.weight = nn.Parameter(weight, requires_grad=False)

    log.info(f"One-Hot Embedding  →  shape: {weight.shape}  trainable: False")
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# 2. Word2Vec Embedding  (trained on IMDB training corpus)
# ─────────────────────────────────────────────────────────────────────────────

def train_word2vec(
    tokenized_train: list[list[str]],
    dim:    int = W2V_DIM,
    window: int = 5,
    min_count: int = 2,
    workers:   int = 4,
    epochs:    int = 10,
) -> "gensim.models.Word2Vec":
    """
    Train a skip-gram Word2Vec model on the IMDB training corpus.

    Architecture : Skip-gram (sg=1) — predicts context words from a
                   centre word. Better than CBOW for infrequent words,
                   which matters for sentiment-bearing adjectives.
    Window       : 5 — context words on each side of the target.
    Min count    : 2 — ignore tokens appearing fewer than 2 times.
    Dimension    : 100
    """
    from gensim.models import Word2Vec

    log.info("Training Word2Vec on IMDB training corpus …")
    log.info(f"  Sentences : {len(tokenized_train):,}")
    log.info(f"  Dim={dim}  window={window}  min_count={min_count}  epochs={epochs}")

    model = Word2Vec(
        sentences  = tokenized_train,
        vector_size= dim,
        window     = window,
        min_count  = min_count,
        sg         = 1,          # skip-gram
        workers    = workers,
        epochs     = epochs,
        seed       = 42,
    )
    return model


def get_word2vec_embedding(
    vocab:           "Vocabulary",
    tokenized_train: list[list[str]] | None = None,
    trainable:       bool = True,
    force:           bool = False,
) -> nn.Embedding:
    """
    Build a Word2Vec embedding matrix aligned to our Vocabulary.

    For tokens present in the Word2Vec model: use the learned vector.
    For tokens absent (OOV relative to Word2Vec): random init N(0, 0.01).
    For <PAD> (index 0): zero vector, frozen.

    Args:
        vocab           : our Vocabulary object
        tokenized_train : list of token lists from training set
                          (required if matrix not cached or force=True)
        trainable       : whether to fine-tune embeddings during RNN training
        force           : retrain Word2Vec even if cache exists
    """
    if W2V_CACHE.exists() and not force:
        log.info(f"Loading cached Word2Vec matrix from {W2V_CACHE} …")
        matrix = np.load(W2V_CACHE)
    else:
        if tokenized_train is None:
            raise ValueError("tokenized_train required to build Word2Vec matrix.")

        w2v_model = train_word2vec(tokenized_train)
        matrix    = _build_matrix(vocab, w2v_model.wv, W2V_DIM)
        np.save(W2V_CACHE, matrix)
        log.info(f"Word2Vec matrix cached → {W2V_CACHE}")

    emb = _matrix_to_embedding(matrix, trainable)
    log.info(f"Word2Vec Embedding  →  shape: {matrix.shape}  "
             f"trainable: {trainable}")
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# 3. GloVe Embedding  (pre-trained, Stanford NLP)
# ─────────────────────────────────────────────────────────────────────────────

def load_glove_vectors(glove_path: Path = GLOVE_PATH) -> dict[str, np.ndarray]:
    """
    Parse the GloVe text file into a {word: vector} dictionary.
    Uses GloVe 6B 100d: trained on 6 billion tokens (Wikipedia + Gigaword),
    840B unique tokens total, 400k vocabulary, 100-dimensional vectors.
    """
    log.info(f"Loading GloVe vectors from {glove_path} …")
    glove = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in progress(f, desc="  reading GloVe"):
            parts  = line.rstrip().split(" ")
            word   = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            glove[word] = vector
    log.info(f"  GloVe vocab loaded: {len(glove):,} vectors")
    return glove


def get_glove_embedding(
    vocab:     "Vocabulary",
    trainable: bool = True,
    force:     bool = False,
) -> nn.Embedding:
    """
    Build a GloVe embedding matrix aligned to our Vocabulary.

    For tokens present in GloVe    : use the pre-trained 100d vector.
    For tokens absent from GloVe   : random init N(0, 0.01).
    For <PAD> (index 0)            : zero vector, frozen during training.

    Args:
        vocab     : our Vocabulary object
        trainable : fine-tune GloVe vectors during RNN training (recommended)
        force     : reload from .txt file even if numpy cache exists
    """
    if GLOVE_CACHE.exists() and not force:
        log.info(f"Loading cached GloVe matrix from {GLOVE_CACHE} …")
        matrix = np.load(GLOVE_CACHE)
    else:
        glove_vecs = load_glove_vectors()
        matrix     = _build_matrix(vocab, glove_vecs, GLOVE_DIM)
        np.save(GLOVE_CACHE, matrix)
        log.info(f"GloVe matrix cached → {GLOVE_CACHE}")

    emb = _matrix_to_embedding(matrix, trainable)
    log.info(f"GloVe Embedding     →  shape: {matrix.shape}  "
             f"trainable: {trainable}")
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_matrix(
    vocab:      "Vocabulary",
    vectors,                        # gensim KeyedVectors or dict
    dim:        int,
) -> np.ndarray:
    """
    Construct a (vocab_size × dim) numpy matrix.
    Row i = embedding for vocab.idx2word[i].
    """
    vocab_size = len(vocab)
    matrix     = np.random.normal(0, 0.01, (vocab_size, dim)).astype(np.float32)
    matrix[0]  = 0.0               # <PAD> → zero vector

    hits = 0
    for word, idx in vocab.word2idx.items():
        try:
            # works for both gensim KeyedVectors and plain dict
            vec = vectors[word]
            matrix[idx] = vec
            hits += 1
        except KeyError:
            pass                    # keep random init for OOV

    coverage = hits / vocab_size * 100
    log.info(f"  Embedding coverage : {hits:,}/{vocab_size:,} ({coverage:.1f}%)")
    return matrix


def _matrix_to_embedding(matrix: np.ndarray, trainable: bool) -> nn.Embedding:
    """Convert numpy matrix to nn.Embedding with correct grad settings."""
    vocab_size, dim = matrix.shape
    weight  = torch.tensor(matrix, dtype=torch.float32)
    emb     = nn.Embedding(vocab_size, dim, padding_idx=0)
    emb.weight = nn.Parameter(weight, requires_grad=trainable)
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# Factory — returns (embedding_layer, embed_dim)
# ─────────────────────────────────────────────────────────────────────────────

def get_embedding(
    strategy:        str,
    vocab:           "Vocabulary",
    tokenized_train: list[list[str]] | None = None,
    trainable:       bool = True,
) -> tuple[nn.Embedding, int]:
    """
    Unified factory.

    Args:
        strategy        : "onehot" | "word2vec" | "glove"
        vocab           : fitted Vocabulary object
        tokenized_train : needed only for word2vec
        trainable       : allow gradient updates on embedding weights

    Returns:
        (nn.Embedding, embed_dim)
    """
    strategy = strategy.lower()

    if strategy == "onehot":
        emb = get_onehot_embedding(len(vocab))
        return emb, len(vocab)                  # dim = vocab_size

    elif strategy == "word2vec":
        emb = get_word2vec_embedding(vocab, tokenized_train, trainable)
        return emb, W2V_DIM

    elif strategy == "glove":
        emb = get_glove_embedding(vocab, trainable)
        return emb, GLOVE_DIM

    else:
        raise ValueError(f"Unknown embedding strategy: '{strategy}'. "
                         f"Choose from: onehot, word2vec, glove")