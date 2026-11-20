"""
Error analysis for PAWS pair models: CV plots, ROPE, length/overlap features, bins, gotcha table, BDA.

Reuses:
- part1.baseline.analysis box plot / ROPE comparisons
- part1.baseline.dataset_analysis overlap definition (Jaccard on lowercased wordpunct)
- part2.analysis.fit_bda_model / one_hot
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from tqdm.auto import tqdm

from part1.baseline import analysis as baseline_analysis
from part2 import analysis as part2_analysis

from . import config, embeddings, experiment


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


def accuracy_by_bin(
    df: pd.DataFrame, bin_col: str, correct_col: str = "correct", item_col: str = "item"
) -> pd.DataFrame:
    """Mean accuracy over predictions and number of unique items per bin."""
    grouped = df.groupby(bin_col, observed=True)
    return pd.DataFrame(
        {
            "accuracy": grouped[correct_col].mean(),
            "n": grouped[item_col].nunique(),
        }
    ).reset_index()


def bin_accuracy_frame(
    df: pd.DataFrame,
    bin_col: str,
    system_col: str = "system",
    correct_col: str = "correct",
) -> pd.DataFrame:
    """Accuracy per (bin, system, seed, fold), for part1.baseline.analysis.plot_box_and_points."""
    system_order = list(dict.fromkeys(df[system_col].tolist()))
    grouped = (
        df.groupby([bin_col, system_col, "run", "fold"], observed=True)[correct_col]
        .mean()
        .rename("accuracy")
        .reset_index()
    )
    grouped["system_rank"] = grouped[system_col].map(system_order.index)
    grouped = grouped.sort_values([bin_col, "system_rank", "run", "fold"])
    grouped["model"] = grouped[system_col] + " " + grouped[bin_col].astype(str)
    return grouped[["model", "accuracy", "run", "fold"]].reset_index(drop=True)


def high_overlap_non_paraphrase_error_rate(
    df: pd.DataFrame,
    overlap_col: str = "overlap",
    label_col: str = "label",
    correct_col: str = "correct",
    system_col: str = "system",
    item_col: str = "item",
    quantile: float | None = None,
):
    """
    PAWS gotcha: error rate on high-overlap × non-paraphrase pairs, per system.
    error_rate is averaged over all checkpoint predictions; n counts unique items.

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
        rows.append(
            {
                "system": system,
                "threshold": threshold,
                "quantile": quantile,
                "n": sub[item_col].nunique(),
                "error_rate": 1.0 - sub[correct_col].mean(),
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


def compare_pairs(cv_results, comparisons, rope_size=None):
    """ROPE probabilities for each (A, B) pair in comparisons."""
    if rope_size is None:
        rope_size = config.ROPE_SIZE
    table = []
    for sys_a, sys_b in comparisons:
        posterior = baseline_analysis.posterior_for(cv_results, sys_a, sys_b)
        p_a, p_tie, p_b = posterior.probs(rope_size)
        table.append([sys_a, sys_b, p_a, p_tie, p_b, posterior.mean_diff, posterior.effect_size])
    return pd.DataFrame(
        table,
        columns=["A", "B", "P(A wins)", "P(tie)", "P(B wins)", "mean diff (B-A)", "effect size"],
    )


def trained_run_ids():
    """Ids of RUNS entries with saved results."""
    return [
        run["id"]
        for run in config.RUNS
        if (config.RESULTS_DIR / run["id"] / "summary.json").is_file()
    ]


def load_cv_results(run_ids):
    """Load per-seed fold accuracies in the format of part1.baseline.analysis.make_data_frame."""
    cv_results = OrderedDict()
    for run_id in run_ids:
        cv_results[run_id] = []
        for seed in config.SEEDS:
            seed_metrics_path = config.RESULTS_DIR / run_id / f"seed-{seed}" / "seed_metrics.json"
            if not seed_metrics_path.is_file():
                raise FileNotFoundError(f"Missing {seed_metrics_path}")
            with open(seed_metrics_path, encoding="utf-8") as f:
                seed_metrics = json.load(f)
            cv_results[run_id].append(
                {"accuracies": {fold: metrics["accuracy"] for fold, metrics in seed_metrics.items()}}
            )
    return cv_results


def collect_devtest_predictions(system_ids, devtest_splits, device=None):
    """Devtest predictions of every checkpoint: {system_id: {(seed, fold): predicted labels}}."""
    if device is None:
        device = config.DEVICE

    loaded_embeddings = {}
    predictions = OrderedDict()
    for system in system_ids:
        emb_key = config.run_by_id(system)["emb"]
        if emb_key not in loaded_embeddings:
            loaded_embeddings[emb_key] = embeddings.load_embedding_and_tokenizer(emb_key)
        emb_model, tokenizer = loaded_embeddings[emb_key]

        predictions[system] = OrderedDict()
        for seed, split in zip(config.SEEDS, devtest_splits):
            devtest_inputs = [pair for pair, _label in split["devtest"]]
            for fold in range(config.NUM_FOLDS):
                model_path = config.RESULTS_DIR / system / f"seed-{seed}" / f"fold-{fold}" / "model"
                if not Path(f"{model_path}.pt").is_file():
                    raise FileNotFoundError(f"Missing checkpoint {model_path}.pt")
                model = experiment.load_model(str(model_path), emb_model, device=device)
                logprobs = experiment.predict_logprobs(
                    model,
                    tokenizer,
                    devtest_inputs,
                    batch_size=config.OPT_KWARGS["batch_size"],
                    device=device,
                )
                predictions[system][(seed, fold)] = logprobs.argmax(axis=-1)
    return predictions


def build_analysis_frame(
    devtest_splits,
    system_predictions: dict,
    tokenizer,
    n_length_bins: int | None = None,
    n_overlap_bins: int | None = None,
) -> pd.DataFrame:
    """
    Build a long DataFrame for bin plots and BDA.

    system_predictions: output of collect_devtest_predictions
    """
    if n_length_bins is None:
        n_length_bins = config.NUM_LENGTH_BINS
    if n_overlap_bins is None:
        n_overlap_bins = config.NUM_OVERLAP_BINS

    features_by_seed = {
        seed: build_pair_features(split["devtest"], tokenizer)
        for seed, split in zip(config.SEEDS, devtest_splits)
    }
    system_to_id = {name: index for index, name in enumerate(system_predictions)}

    frames = []
    for system, checkpoint_predictions in system_predictions.items():
        for (seed, fold), predictions in checkpoint_predictions.items():
            run = config.SEEDS.index(seed)
            features = features_by_seed[seed]
            predictions = np.asarray(predictions, dtype=int)
            if len(predictions) != len(features):
                raise ValueError(
                    f"System {system!r} seed {seed} fold {fold}: got {len(predictions)} "
                    f"predictions, expected {len(features)}"
                )
            frame = features.copy()
            frame.insert(0, "system", system)
            frame.insert(1, "system_id", system_to_id[system])
            frame.insert(2, "item", [int(i) for i in devtest_splits[run]["devtest_ids"]])
            frame.insert(3, "run", run)
            frame.insert(4, "seed", seed)
            frame.insert(5, "fold", fold)
            frame["prediction"] = predictions
            frame["correct"] = (predictions == frame["label"].to_numpy()).astype(int)
            frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["avg_len_words_bin"] = pd.qcut(df["avg_len_words"], q=n_length_bins, duplicates="drop")
    df["avg_len_tokens_bin"] = pd.qcut(df["avg_len_tokens"], q=n_length_bins, duplicates="drop")
    df["overlap_bin"] = pd.qcut(df["overlap"], q=n_overlap_bins, duplicates="drop")
    return df


def fit_bda_from_frame(
    df: pd.DataFrame,
    outcome_space,
    feature_cols=("avg_len_words", "avg_len_tokens", "overlap"),
    seed: int = 0,
):
    """
    Fit part2.analysis.fit_bda_model on an analysis frame from build_analysis_frame.
    Features: one-hot of prediction + continuous feature_cols.
    """
    system_names = list(dict.fromkeys(df["system"].tolist()))
    feature_names = [f"pred:{name}" for name in outcome_space.names] + list(feature_cols)

    features = np.hstack(
        [
            part2_analysis.one_hot(df["prediction"].to_numpy(), len(outcome_space)),
            df[list(feature_cols)].to_numpy(dtype=float),
        ]
    )

    mcmc = part2_analysis.fit_bda_model(
        num_systems=len(system_names),
        sys_id=df["system_id"].to_numpy(),
        features=features,
        correctness=df["correct"].to_numpy(),
        seed=seed,
        num_warmup=config.BDA_NUM_WARMUP,
        num_samples=config.BDA_NUM_SAMPLES,
        num_chains=config.BDA_NUM_CHAINS,
    )
    return mcmc, system_names, feature_names
