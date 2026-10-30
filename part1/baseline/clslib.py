import numpy as np
from tokenizers import Tokenizer


class NamedClasses:
    """
    This is a helper class that wraps around a list of class names and provides 
    functions to map from names (str) to integers.
    """

    def __init__(self, names: list):
        self.names = tuple(names)        
        self._str2int = {name: k for k, name in enumerate(names)}
    
    def index(self, name: str):
        """Convert name to 0-based integer (0-based because Python lists are 0-based)"""
        return self._str2int[name]  # if the name is not recognised, this will raise a ValueError

    def name(self, index: int):
        """Convert an 0-based integer to a class name"""
        return self.names[index]  # if the index is out of bounds, this will raise a IndexError        

    def __len__(self):
        """Size of label space"""
        return len(self.names)

    def __iter__(self):
        """Iterate over names"""
        return iter(self.names)

    def items(self):
        """Iterates over pairs of the kind (index, name) for all categories in the space"""
        return enumerate(self.names)

    def __str__(self):
        return ", ".join(f"{i}: {n}" for i, n in self.items())

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.names)})"


class Encoder:

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        raise NotImplementedError("Concrete implementations will be discussed later")

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        raise NotImplementedError("Implement me in a sub-class")


class TokenIndicator(Encoder):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        unique_token_ids = list(set(self.tokenizer.encode(x).ids))

        # A 0 vector with the same length as the vocabulary except in positions corresponding 
        # to tokens appearing in the document
        vector = np.zeros(len(self))
        vector[unique_token_ids] = 1
        return vector


class BagOfWords(Encoder):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        token_ids = self.tokenizer.encode(x).ids

        return np.bincount(token_ids, minlength=len(self)).astype(float)


PAIR_MODES = ('stack_diff_prod', 'overlap', 'stack', 'concat_text')


class PairEncoder(Encoder):
    """
    Represents a sentence pair x = (s1, s2) as one vector, using a base encoder for single sentences.

    Pair modes:
        - stack_diff_prod: [h(s1), h(s2), |h(s1) - h(s2)|, h(s1) * h(s2)], the combination used by the LSTM classifier
        - overlap: [min(h(s1), h(s2)), |h(s1) - h(s2)|], what is in both sentences followed by what is in only one
            (for token indicators, min(h(s1), h(s2)) is h(s1) * h(s2))
        - stack: [h(s1), h(s2)]
        - concat_text: h(s1 + " " + s2)
    """

    def __init__(self, encoder: Encoder, mode: str):
        """
        encoder: an Encoder for single texts (such as TokenIndicator or BagOfWords)
        mode: one of PAIR_MODES
        """
        if mode not in PAIR_MODES:
            raise ValueError(f"Unknown pair mode={mode}, choose from {PAIR_MODES}")
        self.encoder = encoder
        self.mode = mode

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        if self.mode == 'concat_text':
            return len(self.encoder)
        if self.mode == 'stack_diff_prod':
            return 4 * len(self.encoder)
        return 2 * len(self.encoder)

    def __call__(self, x: tuple) -> 'np.array':
        """Represent the pair x = (s1, s2) as a D-dimensional vector of real numbers"""
        s1, s2 = x
        if self.mode == 'concat_text':
            return self.encoder(s1 + " " + s2)

        h1 = self.encoder(s1)
        h2 = self.encoder(s2)
        if self.mode == 'stack_diff_prod':
            return np.concatenate([h1, h2, np.abs(h1 - h2), h1 * h2])
        if self.mode == 'stack':
            return np.concatenate([h1, h2])
        return np.concatenate([np.minimum(h1, h2), np.abs(h1 - h2)])


def majority_vote(predictions: 'np.array', seed=0) -> 'np.array':
    """
    Combine the decisions of M models for N inputs by majority vote, breaking ties at random 
    (the same rule as part1.clslib.MajorityVoteClassifier, but for all inputs at once).

    predictions: integer array with shape [M, N], one row of predicted labels per model
    seed: we break ties at random
    """
    rng = np.random.RandomState(seed)
    votes = []
    for candidates in predictions.T:
        # count the M decisions for this input
        values, counts = np.unique(candidates, return_counts=True)
        # break ties at random
        votes.append(rng.choice(values[counts == counts.max()]))
    return np.array(votes)
