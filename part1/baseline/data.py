import numpy as np
import itertools
import random
from datasets import load_dataset

from . import clslib


paws_outcome_space = clslib.NamedClasses(['not paraphrase', 'paraphrase'])


def download_paws():
    """
    Returns the labeled (labeled_final) and unlabeled (unlabeled_final) PAWS-Wiki collections.
    """
    labeled = load_dataset("google-research-datasets/paws", "labeled_final")
    unlabeled = load_dataset("google-research-datasets/paws", "unlabeled_final")
    return labeled, unlabeled


def to_pairs(split):
    """
    Returns a list of labelled sentence pairs ((sentence1, sentence2), label) for one split of the dataset.
    """
    return [((s1, s2), int(label)) for s1, s2, label in zip(split["sentence1"], split["sentence2"], split["label"])]


def sentences_of(pairs):
    """
    Returns the sentences in a list of labelled pairs, in order (sentence1, sentence2 of the first pair, and so on).
    """
    return [s for (s1, s2), label in pairs for s in (s1, s2)]


def unlabeled_corpus(unlabeled):
    """
    Returns the sentences (sentence1 and sentence2) of the unlabeled training split, used to train tokenizers.
    """
    return list(unlabeled["train"]["sentence1"]) + list(unlabeled["train"]["sentence2"])


def cross_validation_split(pairs, k, seed=0):
    """
    Shuffles a list of labelled texts and splits the result into k folds.
    
    pairs: lists of tuples (text, label)
    k: number of folds
    seed: for reproducible shuffling
    """
    
    # Shuffle the pairs
    random.seed(seed)
    pairs = list(pairs)
    ids = list(range(len(pairs)))
    random.shuffle(ids)

    # Split without coercing (text, int) tuples into a string array
    ids_parts = np.array_split(np.array(ids, dtype=object), k)
    
    # Create a dictionary to store the training and validation sets for each fold
    cross_validation_sets = {}
    for i in range(k):
        validation_set = [pairs[j] for j in ids_parts[i]] 
        training_ids = list(itertools.chain.from_iterable(part for k, part in enumerate(ids_parts) if k != i))
        training_set = [pairs[j] for j in training_ids]
        cross_validation_sets[i] = {
            "training": training_set, 
            "validation": validation_set, 
            'training_ids': training_ids,   # these are not often needed, but we return it just in case
            'validation_ids': ids_parts[i]  # these are not often needed, but we return it just in case
        }

    return cross_validation_sets
