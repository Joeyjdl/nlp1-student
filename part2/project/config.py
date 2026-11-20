"""
Settings for PAWS paraphrase experiments (part2 project).

Change values here and re-run training or analysis. 
"""
from collections import OrderedDict
from pathlib import Path

import torch

# Paths (relative to this package directory unless noted)
PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
RESULTS_DIR = PACKAGE_DIR / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cross-validation
SEEDS = [0, 3, 5]
NUM_FOLDS = 5

# Devtest held out from the official PAWS test split (for analysis only)
DEVTEST_SIZE = 200  # 100 * 2 classes

# Pair combination (same as part1 DualLSTM / stack_diff_prod)
PAIR_MODE = "stack_diff_prod"

# Shared tokenizer for all embedding models below (uncased word vectors)
TOKENIZER = {
    "kind": "basic",
    "lowercase": True,
    "remove_punct": False,
    "remove_stopwords": False,
    "language": "english",
}

# Embedding tables. Keys are referenced by RUNS["emb"].
# Download URLs are explicit; loaders must not substitute another file or dim.
EMBEDDINGS = OrderedDict([
    ("fasttext300", {
        "family": "fasttext",
        "url": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip",
        "file": "wiki-news-300d-1M.vec.zip",
        "dim": 300,
        "skip_first_line": True,
    }),
    ("fasttext300_subword", {
        "family": "fasttext",
        "url": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip",
        "file": "wiki-news-300d-1M-subword.vec.zip",
        "dim": 300,
        "skip_first_line": True,
    }),
    ("glove300", {
        "family": "glove",
        "url": "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
        "archive": "glove.6B.zip",
        "file": "glove.6B.300d.txt",
        "dim": 300,
        "skip_first_line": False,
    }),
    ("glove50", {
        "family": "glove",
        "url": "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
        "archive": "glove.6B.zip",
        "file": "glove.6B.50d.txt",
        "dim": 50,
        "skip_first_line": False,
    }),
])

# Sentence encoders. Keys are referenced by RUNS["enc"].
ENCODERS = OrderedDict([
    ("embed-and-pool", {
        "enc_kwargs": {
            "freeze_embeddings": True,
            "average_pool": True,
        },
    }),
    ("lstm", {
        "enc_kwargs": {
            "freeze_embeddings": True,
            "hidden_size": 128,
            "word_dropout": 0.0,
        },
    }),
    ("bilstm", {
        "enc_kwargs": {
            "freeze_embeddings": True,
            "hidden_size": 128,
            "word_dropout": 0.0,
        },
    }),
    ("gru", {
        "enc_kwargs": {
            "freeze_embeddings": True,
            "hidden_size": 128,
            "word_dropout": 0.0,
        },
    }),
])

REGRESSOR_KWARGS = {
    "hidden_size1": 50,
    "hidden_size2": 50,
    "dropout": 0.1,
}

OPT_KWARGS = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "num_epochs": 20,
}

# Analysis
OVERLAP_HIGH_QUANTILE = 0.75  # high-overlap bin = overlap >= this quantile on analysis set
NUM_LENGTH_BINS = 5
NUM_OVERLAP_BINS = 5
BDA_NUM_WARMUP = 1000
BDA_NUM_SAMPLES = 1000
BDA_NUM_CHAINS = 4
ROPE_SIZE = 0.005
ROPE_COMPARISONS = [
    ("bilstm_ft300", "lstm_ft300"),
    ("gru_ft300", "lstm_ft300"),
    ("lstm_ft300", "lstm_glove300"),
    ("pool_ft300", "lstm_ft300"),
    ("lstm_glove50", "lstm_glove300"),
    ("lstm_ft300", "lstm_ft_subword"),
]

# Which training runs to execute. Flip enabled to include extras later.
RUNS = [
    # must-have
    {"id": "pool_ft300", "enc": "embed-and-pool", "emb": "fasttext300", "enabled": True},
    {"id": "lstm_ft300", "enc": "lstm", "emb": "fasttext300", "enabled": True},
    {"id": "lstm_glove300", "enc": "lstm", "emb": "glove300", "enabled": True},
    {"id": "bilstm_ft300", "enc": "bilstm", "emb": "fasttext300", "enabled": True},
    {"id": "bilstm_glove300", "enc": "bilstm", "emb": "glove300", "enabled": True},
    # extras (wired, not run)
    {"id": "pool_glove300", "enc": "embed-and-pool", "emb": "glove300", "enabled": False},
    {"id": "pool_glove50", "enc": "embed-and-pool", "emb": "glove50", "enabled": False},
    {"id": "lstm_glove50", "enc": "lstm", "emb": "glove50", "enabled": False},
    {"id": "bilstm_glove50", "enc": "bilstm", "emb": "glove50", "enabled": False},
    {"id": "lstm_ft_subword", "enc": "lstm", "emb": "fasttext300_subword", "enabled": False},
    {"id": "bilstm_ft_subword", "enc": "bilstm", "emb": "fasttext300_subword", "enabled": False},
    {"id": "gru_ft300", "enc": "gru", "emb": "fasttext300", "enabled": False},
    {"id": "gru_glove300", "enc": "gru", "emb": "glove300", "enabled": False},
]


def enabled_runs():
    """Return RUNS entries with enabled=True."""
    return [run for run in RUNS if run["enabled"]]


def run_by_id(run_id: str):
    """Return the RUNS entry with the given id, or raise KeyError."""
    for run in RUNS:
        if run["id"] == run_id:
            return run
    raise KeyError(f"Unknown run id: {run_id}")
