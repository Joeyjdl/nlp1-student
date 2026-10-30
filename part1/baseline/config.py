"""
All settings for the baseline experiments. Change values here and re-run the notebook.
"""
import torch
from collections import OrderedDict

# where models, tokenizers and results are saved (relative to the notebook)
RESULTS_FOLDER = "results"

# use the GPU when available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cross validation: one run of NUM_FOLDS-fold cross validation per seed
SEEDS = [0, 3, 5]
NUM_FOLDS = 5

# BPE tokenizers: one tokenizer is trained per (cleanup variant, vocab size)
BPE_VOCAB_SIZES = [1000]
BPE_MIN_FREQUENCY = 2

# cleanup variants: name -> list of normalizer steps (applied in order)
# available steps: "NFC", "NFD", "NFKC", "NFKD", "StripAccents", "Lowercase"
# StripAccents only removes accents after NFD/NFKD (NFC/NFKC keep accented characters composed)
CLEANUP_VARIANTS = OrderedDict([
    ("raw", []),
    ("nfkd_strip_lower", ["NFKD", "StripAccents", "Lowercase"]),
])

# variants shown in the cleanup comparison section (qualitative and quantitative)
CLEANUP_COMPARISON = OrderedDict([
    ("raw", []),
    ("lower", ["Lowercase"]),
    ("nfkc", ["NFKC"]),
    ("nfkc_lower", ["NFKC", "Lowercase"]),
    ("nfkd_strip", ["NFKD", "StripAccents"]),
    ("nfkd_strip_lower", ["NFKD", "StripAccents", "Lowercase"]),
])
CLEANUP_NUM_EXAMPLES = 5

# encoders: 'indicator' (token present or not) and/or 'bow' (token counts)
ENCODERS = ['indicator', 'bow']

# how a pair (s1, s2) becomes one vector: 'stack_diff_prod', 'overlap', 'stack' and/or 'concat_text'
PAIR_MODES = ['overlap']

# models: 'linear' (log-linear softmax regression) and/or 'mlp' (one hidden layer)
MODELS = ['linear', 'mlp']

# training arguments per model (passed to smlp.train_neural_network_model)
MODEL_ARGS = {
    'mlp': {
        'hidden_dim': 128,
        'dropout': 0.1,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_epochs': 10,
    },
    'linear': {
        'batch_size': 64,
        'learning_rate': 1e-3,
        'weight_decay': 0.0,
        'num_epochs': 10,
        'C': 1.0,  # inverse L2 strength, as in sklearn's LogisticRegression
    },
}

# separate wordpunct run (sparse token indicators over a word vocabulary)
WORDPUNCT_CLEANUP = "raw"  # a key of CLEANUP_COMPARISON
WORDPUNCT_MIN_FREQUENCY = 5
WORDPUNCT_PAIR_MODE = 'overlap'
WORDPUNCT_MODELS = ['linear']  # add 'mlp' to also run the MLP

# Bayesian comparison
ROPE_SIZE = 0.005
# pairs (A, B) of system names to plot posteriors for (names as printed by the experiment section)
ROPE_COMPARISONS = [
    ("linear_indicator-enc_overlap_bpe-raw-vs1000-mf2", "linear_indicator-enc_overlap_bpe-nfkd_strip_lower-vs1000-mf2"),
    ("linear_indicator-enc_overlap_bpe-raw-vs1000-mf2", "linear_bow-enc_overlap_bpe-raw-vs1000-mf2"),
    ("linear_indicator-enc_overlap_bpe-raw-vs1000-mf2", "mlp_indicator-enc_overlap_bpe-raw-vs1000-mf2"),
    ("linear_indicator-enc_overlap_bpe-raw-vs1000-mf2", "linear_indicator-enc_overlap_wordpunct-raw-mf5"),
]
