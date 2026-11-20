"""
Download and load embedding tables + matching tokenizer from project config.

No fallbacks: missing files, missing archive members, or dim mismatches raise.
Reuses part2.clslib.PretrainedEmbeddings and BasicTokenizer.
"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from part2 import clslib

from . import config


def embedding_spec(emb_key: str) -> dict:
    """Return the EMBEDDINGS entry for emb_key, or raise KeyError."""
    if emb_key not in config.EMBEDDINGS:
        raise KeyError(
            f"Unknown embedding key {emb_key!r}. "
            f"Known keys: {list(config.EMBEDDINGS)}"
        )
    return config.EMBEDDINGS[emb_key]


def _download(url: str, destination: Path) -> None:
    """Download url to destination. Raises if the HTTP request fails."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {destination}")
    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"Failed to download {url}") from exc


def _resolve_load_path(spec: dict) -> Path:
    """
    Ensure the embedding file exists under DATA_DIR and return the path
    that PretrainedEmbeddings should open.

    - fasttext: zip (or file) named by spec['file'] is downloaded from spec['url']
      and passed through (part2 supports single-member zips).
    - glove: archive named by spec['archive'] is downloaded; the member
      spec['file'] is extracted to DATA_DIR and that .txt path is returned.
    """
    data_dir = config.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    if "archive" in spec:
        archive_path = data_dir / spec["archive"]
        member_name = spec["file"]
        extract_path = data_dir / member_name

        if not archive_path.is_file():
            _download(spec["url"], archive_path)

        if not extract_path.is_file():
            with zipfile.ZipFile(archive_path) as zf:
                if member_name not in zf.namelist():
                    raise FileNotFoundError(
                        f"Archive {archive_path} has no member {member_name!r}. "
                        f"Members: {zf.namelist()}"
                    )
                print(f"Extracting {member_name} from {archive_path}")
                zf.extract(member_name, path=data_dir)

        if not extract_path.is_file():
            raise FileNotFoundError(
                f"Expected extracted file {extract_path} after unzip, not found."
            )
        return extract_path

    file_path = data_dir / spec["file"]
    if not file_path.is_file():
        _download(spec["url"], file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Embedding file not found: {file_path}")
    return file_path


def ensure_embedding_file(emb_key: str) -> Path:
    """Download/extract if needed; return the path passed to PretrainedEmbeddings."""
    return _resolve_load_path(embedding_spec(emb_key))


def load_embedding_model(emb_key: str) -> clslib.PretrainedEmbeddings:
    """
    Load PretrainedEmbeddings for emb_key using config dim and skip_first_line.
    Raises if the loaded matrix width does not equal config dim, or vocab is empty.
    """
    spec = embedding_spec(emb_key)
    path = ensure_embedding_file(emb_key)
    expected_dim = int(spec["dim"])

    emb_model = clslib.PretrainedEmbeddings(
        path=str(path),
        num_dims=expected_dim,
        skip_first_line=bool(spec["skip_first_line"]),
    )

    if emb_model.num_dims != expected_dim:
        raise ValueError(
            f"Embedding {emb_key!r}: config dim={expected_dim}, "
            f"loaded num_dims={emb_model.num_dims}"
        )
    if emb_model.vectors.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding {emb_key!r}: config dim={expected_dim}, "
            f"vectors.shape={emb_model.vectors.shape}"
        )
    # pad + unk only would mean nothing was loaded for this dim
    if emb_model.vectors.shape[0] <= 2:
        raise ValueError(
            f"Embedding {emb_key!r}: no word vectors loaded from {path} "
            f"with dim={expected_dim}"
        )

    return emb_model


def make_tokenizer(emb_model: clslib.PretrainedEmbeddings) -> clslib.BasicTokenizer:
    """
    Build BasicTokenizer from config.TOKENIZER, bound to emb_model.vocab.
    Raises if TOKENIZER kind is not 'basic'.
    """
    tok = config.TOKENIZER
    kind = tok.get("kind")
    if kind != "basic":
        raise ValueError(f"Unsupported TOKENIZER kind {kind!r}; expected 'basic'")

    return clslib.BasicTokenizer(
        lowercase=bool(tok["lowercase"]),
        remove_punct=bool(tok["remove_punct"]),
        remove_stopwords=bool(tok["remove_stopwords"]),
        language=tok.get("language", "english"),
        vocab=emb_model.vocab,
    )


def load_embedding_and_tokenizer(emb_key: str):
    """
    Convenience: (emb_model, tokenizer) for emb_key.
    Tokenizer vocab is the embedding vocab (required for ids()).
    """
    emb_model = load_embedding_model(emb_key)
    tokenizer = make_tokenizer(emb_model)
    return emb_model, tokenizer
