"""
Error analysis for PAWS pair models: length/overlap features, bins, gotcha table, BDA.

Reuses:
- part1.baseline.dataset_analysis overlap definition (Jaccard on lowercased wordpunct)
- part2.analysis.fit_bda_model / one_hot
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import wordpunct_tokenize
from tqdm.auto import tqdm

from part2 import analysis as part2_analysis

from . import config


def avg_len_words(s1: str, s2: str) -> float:
    """Average sentence length in wordpunct tokens (words)."""
    return 0.5 * (len(wordpunct_tokenize(s1)) + len(wordpunct_tokenize(s2)))


def avg_len_tokens(s1: str, s2: str, tokenizer) -> float:
    """Average sentence length in embedding-tokenizer tokens."""
    return 0.5 * (len(tokenizer.tokens(s1)) + len(tokenizer.tokens(s2)))


def word_overlap(s1: str, s2: str) -> float:
    """
    Jaccard overlap of lowercased wordpunct token sets.
    Same definition as part1.baseline.dataset_analysis.word_overlap.
    """
    w1 = set(t.lower() for t in wordpunct_tokenize(s1))
    w2 = set(t.lower() for t in wordpunct_tokenize(s2))
    union = w1 | w2
    if not union:
        return 0.0
    return len(w1 & w2) / len(union)


def pair_feature_row(s1: str, s2: str, label: int, tokenizer) -> dict:
    """Feature dict for one labelled pair."""
    return {
        "avg_len_words": avg_len_words(s1, s2),
        "avg_len_tokens": avg_len_tokens(s1, s2, tokenizer),
        "overlap": word_overlap(s1, s2),
        "label": int(label),
    }


def build_pair_features(pairs, tokenizer) -> pd.DataFrame:
    """
    pairs: list of ((s1, s2), label)
    Returns a DataFrame with avg_len_words, avg_len_tokens, overlap, label.
    """
    rows = []
    for (s1, s2), label in tqdm(pairs, desc="pair features"):
        rows.append(pair_feature_row(s1, s2, label, tokenizer))
    return pd.DataFrame(rows)


def add_bins(df: pd.DataFrame, column: str, n_bins: int, bin_col: str | None = None) -> pd.DataFrame:
    """
    Add a categorical bin column via pd.qcut (quantile bins).
    Drops duplicate bin edges if needed (qcut duplicates='drop').
    """
    out = df.copy()
    name = bin_col or f"{column}_bin"
    out[name] = pd.qcut(out[column], q=n_bins, duplicates="drop")
    return out


def accuracy_by_bin(df: pd.DataFrame, bin_col: str, correct_col: str = "correct") -> pd.DataFrame:
    """Mean accuracy (and count) per bin."""
    grouped = df.groupby(bin_col, observed=True)[correct_col]
    return pd.DataFrame(
        {
            "accuracy": grouped.mean(),
            "n": grouped.size(),
        }
    ).reset_index()


def plot_accuracy_vs_bins(
    df: pd.DataFrame,
    bin_col: str,
    system_col: str = "system",
    correct_col: str = "correct",
    title: str | None = None,
):
    """
    Line/bar plot of accuracy vs feature bins for each system in df.
    df must contain bin_col, system_col, correct_col.
    """
    rows = []
    for system, sub in df.groupby(system_col):
        stats = accuracy_by_bin(sub, bin_col, correct_col)
        stats[system_col] = system
        rows.append(stats)
    plot_df = pd.concat(rows, ignore_index=True)
    plot_df[bin_col] = plot_df[bin_col].astype(str)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=plot_df, x=bin_col, y="accuracy", hue=system_col, ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(bin_col)
    ax.set_ylabel("accuracy")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    plt.show()
    return plot_df


def high_overlap_non_paraphrase_error_rate(
    df: pd.DataFrame,
    overlap_col: str = "overlap",
    label_col: str = "label",
    correct_col: str = "correct",
    system_col: str = "system",
    quantile: float | None = None,
):
    """
    PAWS gotcha: error rate on high-overlap × non-paraphrase pairs, per system.

    High overlap = overlap >= quantile of overlap on the analysis set
    (default config.OVERLAP_HIGH_QUANTILE). Non-paraphrase = label == 0.
    """
    if quantile is None:
        quantile = config.OVERLAP_HIGH_QUANTILE
    threshold = float(df[overlap_col].quantile(quantile))
    mask = (df[overlap_col] >= threshold) & (df[label_col] == 0)
    subset = df.loc[mask]
    rows = []
    for system, sub in subset.groupby(system_col):
        n = len(sub)
        errors = int((~sub[correct_col].astype(bool)).sum())
        rows.append(
            {
                "system": system,
                "threshold": threshold,
                "quantile": quantile,
                "n": n,
                "errors": errors,
                "error_rate": (errors / n) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows), threshold


def length_by_model_table(
    df: pd.DataFrame,
    length_bin_col: str,
    systems=("lstm_ft300", "bilstm_ft300"),
    system_col: str = "system",
    correct_col: str = "correct",
) -> pd.DataFrame:
    """
    Accuracy per length bin for selected systems (e.g. uni-LSTM vs BiLSTM).
    Wide table: one accuracy column per system.
    """
    pieces = []
    for system in systems:
        sub = df[df[system_col] == system]
        if sub.empty:
            raise KeyError(f"No rows for system {system!r} in analysis frame")
        stats = accuracy_by_bin(sub, length_bin_col, correct_col)
        stats = stats.rename(columns={"accuracy": f"acc_{system}", "n": f"n_{system}"})
        pieces.append(stats.set_index(length_bin_col))
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.join(piece, how="outer")
    return out.reset_index()


def build_analysis_frame(
    pairs,
    system_predictions: dict,
    tokenizer,
    n_length_bins: int | None = None,
    n_overlap_bins: int | None = None,
) -> pd.DataFrame:
    """
    Build a long DataFrame for bin plots and BDA.

    pairs: list of ((s1, s2), label) — same order as each prediction list
    system_predictions: {system_name: array/list of predicted labels length N}
    """
    if n_length_bins is None:
        n_length_bins = config.NUM_LENGTH_BINS
    if n_overlap_bins is None:
        n_overlap_bins = config.NUM_OVERLAP_BINS

    features = build_pair_features(pairs, tokenizer)
    n = len(features)
    frames = []
    system_names = list(system_predictions.keys())
    system_to_id = {name: i for i, name in enumerate(system_names)}

    for system, preds in system_predictions.items():
        preds = np.asarray(preds, dtype=int)
        if len(preds) != n:
            raise ValueError(
                f"System {system!r}: got {len(preds)} predictions, expected {n}"
            )
        labels = features["label"].to_numpy()
        frame = features.copy()
        frame["system"] = system
        frame["system_id"] = system_to_id[system]
        frame["prediction"] = preds
        frame["correct"] = (preds == labels).astype(int)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    # Bins from the feature distribution (same edges for all systems: compute on one copy of pairs)
    word_bins = pd.qcut(features["avg_len_words"], q=n_length_bins, duplicates="drop")
    tok_bins = pd.qcut(features["avg_len_tokens"], q=n_length_bins, duplicates="drop")
    ov_bins = pd.qcut(features["overlap"], q=n_overlap_bins, duplicates="drop")
    # map pair index -> bin via repeating features rows
    pair_index = np.tile(np.arange(n), len(system_names))
    df["avg_len_words_bin"] = word_bins.to_numpy()[pair_index]
    df["avg_len_tokens_bin"] = tok_bins.to_numpy()[pair_index]
    df["overlap_bin"] = ov_bins.to_numpy()[pair_index]
    return df


def fit_bda_from_frame(
    df: pd.DataFrame,
    feature_cols=("avg_len_words", "avg_len_tokens", "overlap"),
    include_pred_one_hot: bool = True,
    num_classes: int = 2,
    seed: int = 0,
):
    """
    Fit part2.analysis.fit_bda_model on an analysis frame from build_analysis_frame.
    Features: optional one-hot of prediction + continuous feature_cols.
    """
    system_names = list(dict.fromkeys(df["system"].tolist()))
    num_systems = len(system_names)

    blocks = []
    if include_pred_one_hot:
        blocks.append(part2_analysis.one_hot(df["prediction"].to_numpy(), num_classes))
    cont = df[list(feature_cols)].to_numpy(dtype=float)
    blocks.append(cont)
    features = np.hstack(blocks)

    mcmc = part2_analysis.fit_bda_model(
        num_systems=num_systems,
        sys_id=df["system_id"].to_numpy(),
        features=features,
        correctness=df["correct"].to_numpy(),
        seed=seed,
        num_warmup=config.BDA_NUM_WARMUP,
        num_samples=config.BDA_NUM_SAMPLES,
        num_chains=config.BDA_NUM_CHAINS,
    )
    return mcmc, system_names, feature_cols
