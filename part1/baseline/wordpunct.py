"""
The separate wordpunct run: token indicators over a word vocabulary built with NLTK's wordpunct_tokenize.
The vocabulary is large, so the encoder returns sparse (CSR) rows instead of dense vectors.
"""
import json
import numpy as np
import scipy.sparse
from pathlib import Path
from collections import Counter
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm
from nltk.tokenize import wordpunct_tokenize

from . import clslib
from . import experiment
from . import tokenization


def wordpunct_name(cleanup: str, min_frequency: int):
    """Name used for a wordpunct vocabulary (and in the names of the systems that use it)."""
    return f"wordpunct-{cleanup}-mf{min_frequency}"


def build_vocab(corpus, min_frequency: int, normalizer):
    """
    Return a dict mapping each word that occurs at least min_frequency times in the corpus to a 0-based id.

    corpus: a list of texts (each a string)
    min_frequency: words that occur less often are left out of the vocabulary
    normalizer: cleanup applied before tokenization (see tokenization.make_normalizer)
    """
    counts = Counter()
    for text in tqdm(corpus, desc="Counting words"):
        counts.update(wordpunct_tokenize(normalizer.normalize_str(text)))
    words = sorted(word for word, count in counts.items() if count >= min_frequency)
    return {word: i for i, word in enumerate(words)}


def load_or_build_vocab(corpus, cleanup: str, steps: list, min_frequency: int, results_folder):
    """
    Load a vocabulary from {results_folder}/tokenizers/{name}.json, or build and save it if it is not there.
    The settings are checked against {name}.settings.json (see experiment.check_saved_settings).

    corpus: a list of texts used to build the vocabulary
    cleanup: name of the cleanup variant
    steps: normalizer steps of the cleanup variant
    min_frequency: words that occur less often are left out of the vocabulary
    results_folder: where everything is saved
    """
    name = wordpunct_name(cleanup, min_frequency)
    path = Path(f"{results_folder}/tokenizers/{name}.json")
    experiment.check_saved_settings(
        f"{results_folder}/tokenizers/{name}.settings.json",
        {'steps': steps, 'min_frequency': min_frequency, 'corpus_size': len(corpus)},
    )
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(build_vocab(corpus, min_frequency, tokenization.make_normalizer(steps)), f, ensure_ascii=False)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class WordPunctPairIndicator(clslib.Encoder):
    """
    Represents a sentence pair x = (s1, s2) as a sparse vector of word indicators, using the same pair modes 
    as clslib.PairEncoder (with V the vocabulary size):
        - stack_diff_prod: words of s1 in [0, V), words of s2 in [V, 2V), words in only one sentence in [2V, 3V), 
            words in both sentences in [3V, 4V)
        - overlap: words in both sentences in [0, V), words in only one sentence in [V, 2V)
        - stack: words of s1 in [0, V), words of s2 in [V, 2V)
        - concat_text: words of s1 + " " + s2 in [0, V)
    Words that are not in the vocabulary are ignored.
    """

    def __init__(self, vocab: dict, normalizer, mode: str):
        """
        vocab: dict mapping a word to a 0-based id (see build_vocab)
        normalizer: cleanup applied before tokenization (the same one used to build the vocabulary)
        mode: one of clslib.PAIR_MODES
        """
        if mode not in clslib.PAIR_MODES:
            raise ValueError(f"Unknown pair mode={mode}, choose from {clslib.PAIR_MODES}")
        self.vocab = vocab
        self.normalizer = normalizer
        self.mode = mode

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        if self.mode == 'concat_text':
            return len(self.vocab)
        if self.mode == 'stack_diff_prod':
            return 4 * len(self.vocab)
        return 2 * len(self.vocab)

    def word_ids(self, text: str) -> set:
        """The set of vocabulary ids of the words in text"""
        return {self.vocab[w] for w in wordpunct_tokenize(self.normalizer.normalize_str(text)) if w in self.vocab}

    def __call__(self, x: tuple) -> 'csr_matrix':
        """Represent the pair x = (s1, s2) as a sparse 1 x D row of indicators"""
        s1, s2 = x
        V = len(self.vocab)
        if self.mode == 'concat_text':
            ids = self.word_ids(s1 + " " + s2)
        else:
            ids1, ids2 = self.word_ids(s1), self.word_ids(s2)
            if self.mode == 'stack_diff_prod':
                ids = ids1 | {V + i for i in ids2} | {2 * V + i for i in ids1 ^ ids2} | {3 * V + i for i in ids1 & ids2}
            elif self.mode == 'stack':
                ids = ids1 | {V + i for i in ids2}
            else:
                ids = (ids1 & ids2) | {V + i for i in ids1 ^ ids2}
        ids = sorted(ids)
        return csr_matrix((np.ones(len(ids)), (np.zeros(len(ids), dtype=int), ids)), shape=(1, len(self)))


def vectorize(texts, encoder):
    """Encode inputs with an encoder that returns sparse rows and stack them into a SciPy CSR matrix.

    texts: list of inputs for the encoder
    encoder: an instance of WordPunctPairIndicator
    """
    return scipy.sparse.vstack([encoder(x) for x in texts]).tocsr()
