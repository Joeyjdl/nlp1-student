import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from tabulate import tabulate
from tqdm.auto import tqdm
from nltk.tokenize import (
    word_tokenize,
    wordpunct_tokenize,
    TreebankWordTokenizer,
    TweetTokenizer,
    WhitespaceTokenizer,
)


HEAPS_TOKENIZERS = OrderedDict([
    ("NLTK Word", word_tokenize),
    ("WordPunct", wordpunct_tokenize),
    ("Treebank", TreebankWordTokenizer().tokenize),
    ("Tweet", TweetTokenizer().tokenize),
    ("Whitespace", WhitespaceTokenizer().tokenize),
])


def split_sizes(datasets: dict):
    """
    Return a DataFrame with the number of pairs in every split of every dataset.

    datasets: dict mapping a dataset name to a Hugging Face DatasetDict
    """
    table = []
    for name, dataset in datasets.items():
        for split, data in dataset.items():
            table.append([name, split, len(data)])
    return pd.DataFrame(table, columns=['dataset', 'split', 'pairs'])


def label_counts(dataset, outcome_space):
    """
    Return a DataFrame with the count and fraction of each label in every split.

    dataset: a Hugging Face DatasetDict with a 'label' column
    outcome_space: NamedClasses for the dataset
    """
    table = []
    for split, data in dataset.items():
        labels = np.array(data["label"])
        for k, name in outcome_space.items():
            count = int((labels == k).sum())
            table.append([split, name, count, count / len(labels)])
    return pd.DataFrame(table, columns=['split', 'label', 'count', 'fraction'])


def example_pairs(pairs, outcome_space, num_examples=3, seed=0):
    """
    Return a table-formatted view of a few random pairs per label in a single string ready to be displayed.

    pairs: list of ((sentence1, sentence2), label)
    outcome_space: NamedClasses for the dataset
    num_examples: number of pairs per label
    seed: for reproducible sampling
    """
    rng = np.random.default_rng(seed)
    blocks = []
    for k, name in outcome_space.items():
        ids = [i for i, (x, label) in enumerate(pairs) if label == k]
        table = []
        for i in rng.choice(ids, size=num_examples, replace=False):
            (s1, s2), label = pairs[i]
            table.append(["s1", s1])
            table.append(["s2", s2])
            table.append(["", ""])
        blocks.append(f"Label: {name}\n" + tabulate(table, tablefmt="plain"))
    return "\n\n".join(blocks)


def sentence_lengths(pairs, outcome_space):
    """
    Return a DataFrame with the length in wordpunct tokens of every sentence, with its label.

    pairs: list of ((sentence1, sentence2), label)
    outcome_space: NamedClasses for the dataset
    """
    table = []
    for (s1, s2), label in tqdm(pairs, desc="Lengths"):
        table.append([outcome_space.name(label), 'sentence1', len(wordpunct_tokenize(s1))])
        table.append([outcome_space.name(label), 'sentence2', len(wordpunct_tokenize(s2))])
    return pd.DataFrame(table, columns=['label', 'sentence', 'length'])


def word_overlap(pairs, outcome_space):
    """
    Return a DataFrame with the word overlap of every pair, with its label. 
    Overlap is the Jaccard index of the sets of lowercased wordpunct tokens of sentence1 and sentence2.

    pairs: list of ((sentence1, sentence2), label)
    outcome_space: NamedClasses for the dataset
    """
    table = []
    for (s1, s2), label in tqdm(pairs, desc="Overlap"):
        w1 = set(t.lower() for t in wordpunct_tokenize(s1))
        w2 = set(t.lower() for t in wordpunct_tokenize(s2))
        table.append([outcome_space.name(label), len(w1 & w2) / len(w1 | w2)])
    return pd.DataFrame(table, columns=['label', 'overlap'])


def plot_histogram_per_label(df, column, xlabel, bins=50):
    """
    Plot the distribution of a column of the DataFrame, one histogram per label.

    df: a pandas DataFrame with a 'label' column
    column: the column to plot
    xlabel: label of the horizontal axis
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x=column, hue='label', bins=bins, stat='density', common_norm=False, element='step', ax=ax)
    ax.set(xlabel=xlabel, ylabel='density')
    fig.tight_layout()
    plt.show()


def heaps_tokens(sentences, tokenizers=HEAPS_TOKENIZERS):
    """
    Tokenize a corpus with each tokenizer. Return a dict mapping a tokenizer name to the list of all tokens in the corpus.

    sentences: list of strings
    tokenizers: dict mapping a name to a function from string to list of tokens
    """
    tokens = {name: [] for name in tokenizers}

    for text in tqdm(sentences, desc="Tokenizing"):
        for name, tokenize in tokenizers.items():
            tokens[name].extend(tokenize(text))

    for name, token_list in tokens.items():
        print(f"{name}: {len(token_list)} tokens")
    return tokens


def heaps_curves(tokens):
    """
    Return a dict mapping a tokenizer name to (n, V), with V[i] the number of distinct tokens among the first n[i] tokens.

    tokens: output of heaps_tokens
    """
    curves = {}

    for name, token_list in tqdm(tokens.items(), desc="Counting"):
        seen = set()
        V = np.empty(len(token_list), dtype=np.int32)

        for i, token in enumerate(token_list):
            seen.add(token)
            V[i] = len(seen)

        n = np.arange(1, len(token_list) + 1)

        curves[name] = (n, V)
    return curves


def heaps_fits(curves):
    """
    Fit Heaps' law V(n) = K n^beta (as a line in log-log space). Return a dict mapping a tokenizer name to (K, beta, R^2).

    curves: output of heaps_curves
    """
    fits = {}

    for name, (n, V) in curves.items():
        # Exclude first 1% to avoid early vocabulary growth
        fit_start = max(100, int(len(n) * 0.01))
        mask = n >= fit_start

        log_n = np.log(n[mask])
        log_V = np.log(V[mask])

        beta, log_K = np.polyfit(log_n, log_V, 1)
        K = np.exp(log_K)

        predicted = log_K + beta * log_n
        ss_res = np.sum((log_V - predicted) ** 2)
        ss_tot = np.sum((log_V - np.mean(log_V)) ** 2)
        r2 = 1 - ss_res / ss_tot

        fits[name] = (K, beta, r2)

        print(f"{name:12s}: K={K:.3f}, beta={beta:.4f}, R²={r2:.4f}")
    return fits


def plot_heaps(curves, fits):
    """
    Plot the observed vocabulary growth and the fitted Heaps' law, first all tokenizers together, then one plot per tokenizer.

    curves: output of heaps_curves
    fits: output of heaps_fits
    """
    # === 1. Combined plot ===
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, (n, V) in curves.items():
        K, beta, r2 = fits[name]

        line, = ax.plot(n, V, label=f"{name} (observed)")
        ax.plot(
            n, K * n**beta,
            "--",
            color=line.get_color(),
            alpha=0.8,
            label=f"{name} (fit: β={beta:.3f})"
        )

    ax.set_xlabel("Number of tokens (n)")
    ax.set_ylabel("Vocabulary size V(n)")
    ax.set_title("Heaps' Law — All Tokenizers")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()


    # === 2. Individual plots in a grid ===
    num_tokenizers = len(curves)
    cols = 2
    rows = math.ceil(num_tokenizers / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = np.array(axes).flatten()

    for ax, (name, (n, V)) in zip(axes, curves.items()):
        K, beta, r2 = fits[name]

        ax.plot(n, V, label="Observed", linewidth=2)
        ax.plot(
            n, K * n**beta,
            "--",
            label=f"Fit: K={K:.2f}, β={beta:.3f}, R²={r2:.4f}",
            linewidth=2
        )

        ax.set_xlabel("Number of tokens (n)")
        ax.set_ylabel("Vocabulary size V(n)")
        ax.set_title(name)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    # Hide unused subplots
    for ax in axes[num_tokenizers:]:
        ax.set_visible(False)

    fig.suptitle("Heaps' Law — Individual Tokenizer Fits", fontsize=16)
    plt.tight_layout()
    plt.show()
