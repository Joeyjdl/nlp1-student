from pathlib import Path
from collections import OrderedDict
import pandas as pd
from tabulate import tabulate
from tqdm.auto import tqdm
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace

from . import experiment


NORMALIZER_STEPS = {
    "NFC": normalizers.NFC,
    "NFD": normalizers.NFD,
    "NFKC": normalizers.NFKC,
    "NFKD": normalizers.NFKD,
    "StripAccents": normalizers.StripAccents,
    "Lowercase": normalizers.Lowercase,
}

def make_normalizer(steps: list):
    """
    Build a Hugging Face normalizer that applies the given steps in order.

    steps: list of names from NORMALIZER_STEPS (an empty list leaves text unchanged)
    """
    for step in steps:
        if step not in NORMALIZER_STEPS:
            raise ValueError(f"Unknown normalizer step={step}, choose from {tuple(NORMALIZER_STEPS)}")
    return normalizers.Sequence([NORMALIZER_STEPS[step]() for step in steps])


def train_tokenizer(corpus, vocab_size, min_frequency, normalizer, pretokenizer=Metaspace(), unk_token="[UNK]"):
    """
    corpus: a list of texts (each a string)
    vocab_size: number of distinct BPE tokens
    min_frequency: we only merge token pairs that are frequent enough
    normalizer: cleanup applied to the text before pretokenization (see make_normalizer)
    pretokenizer: a simple tokenizer that we use before BPE tokenization
    
        You probably don't need to change the pretokenizer, unless you are working with
        languages that really don't use any form of blank space 
        (then you could, for example, pretokenize at character level)
    """
    # Create an empty BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token=unk_token)) #unk_token used for characters that cannot be encoded

    # Clean up the text (e.g. lowercase) before anything else
    tokenizer.normalizer = normalizer

    # First split the corpus into words
    tokenizer.pre_tokenizer = pretokenizer

    # Tell BPE how large the vocabulary should be
    trainer = BpeTrainer(
        vocab_size=vocab_size, # The size of the vocabulary to learn.
        min_frequency=min_frequency, # Only perform a merge if the pair occurs at least min_frequency times in the training corpus.
        special_tokens=[unk_token]
    )

    # Learn the vocabulary and merge rules
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    return tokenizer        


def bpe_name(cleanup: str, vocab_size: int, min_frequency: int):
    """Name used for a BPE tokenizer (and in the names of the systems that use it)."""
    return f"bpe-{cleanup}-vs{vocab_size}-mf{min_frequency}"


def load_or_train_bpe(corpus, cleanup: str, steps: list, vocab_size: int, min_frequency: int, results_folder):
    """
    Load a trained BPE tokenizer from {results_folder}/tokenizers/{name}.json, or train and save it if it is not there.
    The settings are checked against {name}.settings.json (see experiment.check_saved_settings).

    corpus: a list of texts used for training
    cleanup: name of the cleanup variant
    steps: normalizer steps of the cleanup variant
    vocab_size: number of distinct BPE tokens
    min_frequency: we only merge token pairs that are frequent enough
    results_folder: where everything is saved
    """
    name = bpe_name(cleanup, vocab_size, min_frequency)
    path = Path(f"{results_folder}/tokenizers/{name}.json")
    experiment.check_saved_settings(
        f"{results_folder}/tokenizers/{name}.settings.json",
        {'steps': steps, 'vocab_size': vocab_size, 'min_frequency': min_frequency, 'corpus_size': len(corpus)},
    )
    if not path.exists():
        tokenizer = train_tokenizer(corpus, vocab_size, min_frequency, make_normalizer(steps))
        tokenizer.save(str(path))
    return Tokenizer.from_file(str(path))


def pick_cleanup_examples(sentences, num_examples):
    """
    Pick sentences that contain non-ASCII characters (these show the effect of unicode normalization and accent stripping).

    sentences: list of strings
    num_examples: how many to return
    """
    return [s for s in sentences if any(ord(ch) > 127 for ch in s)][:num_examples]


def compare_cleanup_qualitative(examples, variants: OrderedDict):
    """
    Return a table-formatted view of each example under each cleanup variant in a single string ready to be displayed.

    examples: list of strings
    variants: dict mapping a variant name to its list of normalizer steps
    """
    blocks = []
    for i, text in enumerate(examples):
        table = [[name, make_normalizer(steps).normalize_str(text)] for name, steps in variants.items()]
        blocks.append(f"Example {i}\n" + tabulate(table, headers=['variant', 'text']))
    return "\n\n".join(blocks)


def compare_cleanup_quantitative(sentences, variants: OrderedDict):
    """
    Return a DataFrame with, per cleanup variant, the number of unique characters and unique words 
    (words are blank-separated, PAWS sentences already have spaces around punctuation).

    sentences: list of strings
    variants: dict mapping a variant name to its list of normalizer steps
    """
    table = []
    for name, steps in tqdm(variants.items(), desc="Cleanup variants"):
        normalizer = make_normalizer(steps)
        chars, words = set(), set()
        for text in sentences:
            clean = normalizer.normalize_str(text)
            chars.update(clean)
            words.update(clean.split())
        table.append([name, " + ".join(steps) if steps else "-", len(chars), len(words)])
    return pd.DataFrame(table, columns=['variant', 'steps', 'unique characters', 'unique words'])
