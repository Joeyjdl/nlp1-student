"""
Train and evaluate PAWS pair classifiers (stack_diff_prod).

Reuses:
- part1.baseline.data for PAWS loading / outcome space
- part1.experiment for CV and devtest splits
- part2.experiment patterns for epoch train / best-val-loss checkpointing
"""
from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from part1 import experiment as part1_experiment
from part1.baseline import data as paws_data
from part2 import clslib as part2_clslib

from . import clslib, config, embeddings


class LabelledPairCorpus(Dataset):
    """List of (token-id sequence s1, token-id sequence s2, label)."""

    def __init__(self, X1, X2, y):
        self.X1 = [torch.tensor(x, dtype=torch.long) for x in X1]
        self.X2 = [torch.tensor(x, dtype=torch.long) for x in X2]
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]


def collate_labelled_pairs(batched_triples, padding_value=0):
    x1_rows, x2_rows, y_rows = zip(*batched_triples)
    x1_batch = pad_sequence(x1_rows, batch_first=True, padding_value=padding_value)
    x2_batch = pad_sequence(x2_rows, batch_first=True, padding_value=padding_value)
    y_batch = torch.stack(y_rows)
    return x1_batch, x2_batch, y_batch


def _tokenize_pairs(pairs, tokenizer):
    """pairs: list of ((s1, s2), label) -> X1 ids, X2 ids, labels."""
    x1, x2, y = [], [], []
    for (s1, s2), label in pairs:
        ids1 = tokenizer.ids(s1)
        ids2 = tokenizer.ids(s2)
        if len(ids1) == 0:
            ids1 = [tokenizer.vocab.default_index]
        if len(ids2) == 0:
            ids2 = [tokenizer.vocab.default_index]
        x1.append(ids1)
        x2.append(ids2)
        y.append(int(label))
    return x1, x2, y


def _classification_metrics(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    per_class = f1_score(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_per_class": [float(v) for v in per_class],
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0)
        ),
    }


def training_one_epoch(model, dataloader, optimizer, criterion, device="cpu"):
    if isinstance(device, str):
        device = torch.device(device)

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x1_batch, x2_batch, y_batch in dataloader:
        x1_batch = x1_batch.to(device)
        x2_batch = x2_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logprobs = model(x1_batch, x2_batch)
        loss = criterion(logprobs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        predictions = logprobs.argmax(dim=1)
        total_correct += (predictions == y_batch).sum().item()
        total_examples += y_batch.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def evaluate(model, dataloader, criterion, num_classes: int, device="cpu"):
    if isinstance(device, str):
        device = torch.device(device)

    model.eval()
    total_loss = 0.0
    total_examples = 0
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for x1_batch, x2_batch, y_batch in dataloader:
            x1_batch = x1_batch.to(device)
            x2_batch = x2_batch.to(device)
            y_batch = y_batch.to(device)

            logprobs = model(x1_batch, x2_batch)
            loss = criterion(logprobs, y_batch)

            total_loss += loss.item() * y_batch.size(0)
            predictions = logprobs.argmax(dim=1)
            total_examples += y_batch.size(0)
            preds_list.extend(predictions.tolist())
            labels_list.extend(y_batch.tolist())

    metrics = _classification_metrics(labels_list, preds_list, num_classes)
    metrics["loss"] = total_loss / total_examples
    metrics["predictions"] = preds_list
    metrics["labels"] = labels_list
    return metrics


def _build_model(enc_type, emb_model, enc_kwargs, regressor_kwargs, num_classes, device):
    pad_id = emb_model.vocab.index(emb_model.pad_token)
    enc_kwargs = dict(enc_kwargs)
    enc_kwargs["padding_idx"] = pad_id

    embedding_matrix = torch.tensor(emb_model.vectors, dtype=torch.float32)
    encoder = clslib.make_sentence_encoder(enc_type, embedding_matrix, enc_kwargs)
    model = clslib.PairStackDiffProdClassifier(
        encoder=encoder,
        num_classes=num_classes,
        **regressor_kwargs,
    )
    return model.to(device), pad_id


def train(
    training_set,
    validation_set,
    tokenizer: part2_clslib.Tokenizer,
    emb_model: part2_clslib.PretrainedEmbeddings,
    num_classes: int,
    enc_type: str,
    save_to="temp",
    enc_kwargs={"freeze_embeddings": True},
    regressor_kwargs={"hidden_size1": 50, "hidden_size2": 50, "dropout": 0.0},
    opt_kwargs={"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "num_epochs": 20},
    device="cpu",
):
    """
    Train for opt_kwargs['num_epochs'] epochs.
    Save the checkpoint with best validation loss to save_to.pt / save_to.json.
    Returns (epoch_metrics_list, best_epoch).
    """
    time_start = time.time()

    tqdm.write("Tokenizing pairs...")
    train_x1, train_x2, train_y = _tokenize_pairs(training_set, tokenizer)
    val_x1, val_x2, val_y = _tokenize_pairs(validation_set, tokenizer)

    time_tokenizing = time.time()
    tqdm.write(f"Tokenization done! Time taken: {time_tokenizing - time_start:.3f} seconds")

    pad_id = emb_model.vocab.index(emb_model.pad_token)

    train_loader = DataLoader(
        LabelledPairCorpus(train_x1, train_x2, train_y),
        batch_size=opt_kwargs["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate_labelled_pairs(batch, padding_value=pad_id),
    )
    val_loader = DataLoader(
        LabelledPairCorpus(val_x1, val_x2, val_y),
        batch_size=opt_kwargs["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: collate_labelled_pairs(batch, padding_value=pad_id),
    )

    time_dataloader = time.time()
    tqdm.write(f"Dataloader done! Time taken: {time_dataloader - time_tokenizing:.3f} seconds")

    if isinstance(device, str):
        device = torch.device(device)

    model, pad_id = _build_model(
        enc_type, emb_model, enc_kwargs, regressor_kwargs, num_classes, device
    )

    args = {
        "enc_type": enc_type,
        "enc_kwargs": {**enc_kwargs, "padding_idx": pad_id},
        "regressor_kwargs": {**regressor_kwargs, "num_classes": num_classes},
        "pair_mode": config.PAIR_MODE,
    }
    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    with open(f"{save_to}.json", "w", encoding="utf-8") as f:
        json.dump(args, f, indent=2, ensure_ascii=False)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_kwargs["learning_rate"],
        weight_decay=opt_kwargs["weight_decay"],
    )

    metrics = []
    best_val_loss = None
    best_epoch = 0

    for epoch in tqdm(range(opt_kwargs["num_epochs"]), desc="epochs", leave=False):
        train_metrics = training_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, num_classes, device)
        metrics.append({"train_metrics": train_metrics, "val_metrics": val_metrics})

        tqdm.write(
            f"Epoch {epoch + 1:02d} | "
            f"train loss: {train_metrics['loss']:.4f} | "
            f"train acc: {train_metrics['accuracy']:.4f} | "
            f"val loss: {val_metrics['loss']:.4f} | "
            f"val acc: {val_metrics['accuracy']:.4f} | "
            f"val macro_f1: {val_metrics['macro_f1']:.4f}"
        )

        if best_val_loss is None or val_metrics["loss"] <= best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), f"{save_to}.pt")
            best_epoch = epoch

    time_training = time.time()
    tqdm.write(f"Training done! Time taken: {time_training - time_dataloader:.3f} seconds")
    tqdm.write(f"Best epoch {best_epoch + 1} (val loss {best_val_loss:.4f})")
    return metrics, best_epoch


def load_model(path, emb_model: part2_clslib.PretrainedEmbeddings, device="cpu"):
    """Reconstruct PairStackDiffProdClassifier from path.json + path.pt."""
    with open(f"{path}.json", "r", encoding="utf-8") as f:
        args = json.load(f)
    if isinstance(device, str):
        device = torch.device(device)

    model, _ = _build_model(
        enc_type=args["enc_type"],
        emb_model=emb_model,
        enc_kwargs=args["enc_kwargs"],
        regressor_kwargs={
            k: v for k, v in args["regressor_kwargs"].items() if k != "num_classes"
        },
        num_classes=args["regressor_kwargs"]["num_classes"],
        device=device,
    )
    state = torch.load(f"{path}.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def cross_validation(
    outcome_space,
    data_splits,
    tokenizer: part2_clslib.Tokenizer,
    emb_model: part2_clslib.PretrainedEmbeddings,
    enc_type: str,
    enc_kwargs,
    regressor_kwargs,
    opt_kwargs,
    device="cpu",
    save_to="tmp",
):
    """
    Train one model per fold. Returns dict with models, predictions, metrics per fold.
    Also writes history.json (full per-epoch metrics) under each fold folder.
    """
    models = OrderedDict()
    fold_metrics = OrderedDict()
    predictions = OrderedDict()

    num_classes = len(outcome_space)
    for fold in tqdm(list(data_splits.keys()), desc="folds"):
        fold_folder = Path(f"{save_to}/fold-{fold}")
        fold_folder.mkdir(parents=True, exist_ok=True)
        save_prefix = str(fold_folder / "model")

        metrics, best_epoch = train(
            training_set=data_splits[fold]["training"],
            validation_set=data_splits[fold]["validation"],
            tokenizer=tokenizer,
            emb_model=emb_model,
            num_classes=num_classes,
            enc_type=enc_type,
            save_to=save_prefix,
            enc_kwargs=enc_kwargs,
            regressor_kwargs=regressor_kwargs,
            opt_kwargs=opt_kwargs,
            device=device,
        )

        with open(fold_folder / "history.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        best_val = metrics[best_epoch]["val_metrics"]
        models[fold] = save_prefix
        fold_metrics[fold] = {
            "best_epoch": best_epoch,
            "accuracy": best_val["accuracy"],
            "f1_per_class": best_val["f1_per_class"],
            "macro_f1": best_val["macro_f1"],
            "loss": best_val["loss"],
        }
        predictions[fold] = {
            "predictions": best_val["predictions"],
            "labels": best_val["labels"],
        }

        with open(fold_folder / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(fold_metrics[fold], f, indent=2)

    return {"models": models, "predictions": predictions, "metrics": fold_metrics}


def predict_logprobs(
    model: clslib.PairStackDiffProdClassifier,
    tokenizer: part2_clslib.Tokenizer,
    pairs_or_texts,
    batch_size: int,
    device="cpu",
):
    """
    pairs_or_texts: list of (s1, s2) sentence pairs (no labels).
    Returns array [N, num_classes] of log-probabilities.
    """
    if isinstance(device, str):
        device = torch.device(device)

    labelled = [((s1, s2), 0) for s1, s2 in pairs_or_texts]
    x1, x2, y = _tokenize_pairs(labelled, tokenizer)
    pad_id = model.pad_idx
    loader = DataLoader(
        LabelledPairCorpus(x1, x2, y),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_labelled_pairs(batch, padding_value=pad_id),
    )

    model.eval()
    logprobs_list = []
    with torch.no_grad():
        for x1_batch, x2_batch, _y in loader:
            x1_batch = x1_batch.to(device)
            x2_batch = x2_batch.to(device)
            logprobs_list.append(model(x1_batch, x2_batch).cpu().numpy())
    return np.concatenate(logprobs_list, axis=0)


def summarize_metrics(fold_metrics_by_seed):
    """
    fold_metrics_by_seed: list of dicts fold -> metrics (one per seed).
    Returns mean/std over all fold runs for accuracy, macro_f1, f1_per_class.
    """
    accs, macros = [], []
    per_class = []
    for seed_metrics in fold_metrics_by_seed:
        for fold_m in seed_metrics.values():
            accs.append(fold_m["accuracy"])
            macros.append(fold_m["macro_f1"])
            per_class.append(fold_m["f1_per_class"])

    per_class = np.asarray(per_class, dtype=float)
    return {
        "n": len(accs),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "macro_f1_mean": float(np.mean(macros)),
        "macro_f1_std": float(np.std(macros, ddof=1)) if len(macros) > 1 else 0.0,
        "f1_per_class_mean": [float(v) for v in per_class.mean(axis=0)],
        "f1_per_class_std": (
            [float(v) for v in per_class.std(axis=0, ddof=1)]
            if len(accs) > 1
            else [0.0] * per_class.shape[1]
        ),
    }


def load_paws_cv_and_devtest():
    """
    Load PAWS labeled_final; build CV pairs (train+validation) and per-seed devtest splits
    from the official test set.
    """
    labeled, _unlabeled = paws_data.download_paws()
    outcome_space = paws_data.paws_outcome_space
    cv_pairs = paws_data.to_pairs(labeled["train"]) + paws_data.to_pairs(labeled["validation"])
    test_pairs = paws_data.to_pairs(labeled["test"])

    cv_splits = []
    devtest_splits = []
    for seed in config.SEEDS:
        cv_splits.append(
            part1_experiment.cross_validation_split(cv_pairs, k=config.NUM_FOLDS, seed=seed)
        )
        devtest_splits.append(
            part1_experiment.split_test_devtest(test_pairs, config.DEVTEST_SIZE, seed=seed)
        )
    return outcome_space, cv_splits, devtest_splits


def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_one(run, outcome_space, cv_splits, device=None, force=False):
    """Run 3-seed x 5-fold CV for one RUNS entry, reusing saved results unless force. Returns summary dict."""
    if device is None:
        device = config.DEVICE

    save_root = config.RESULTS_DIR / run["id"]
    summary_path = save_root / "summary.json"
    if summary_path.is_file() and not force:
        print(f"=== Run {run['id']}: already trained, skipping (use --force to retrain) ===")
        return _read_json(summary_path)

    enc_name = run["enc"]
    emb_key = run["emb"]
    if enc_name not in config.ENCODERS:
        raise KeyError(f"Unknown enc {enc_name!r}")
    if emb_key not in config.EMBEDDINGS:
        raise KeyError(f"Unknown emb {emb_key!r}")
    if config.PAIR_MODE != "stack_diff_prod":
        raise ValueError(f"Only stack_diff_prod is supported, got {config.PAIR_MODE!r}")

    enc_kwargs = dict(config.ENCODERS[enc_name]["enc_kwargs"])
    print(f"=== Run {run['id']}: enc={enc_name} emb={emb_key} ===")
    emb_model, tokenizer = None, None

    seed_metrics = []
    for seed_idx, seed in enumerate(config.SEEDS):
        save_to = str(save_root / f"seed-{seed}")
        seed_metrics_path = Path(save_to) / "seed_metrics.json"
        if seed_metrics_path.is_file() and not force:
            print(f"[{run['id']}] seed {seed}: already trained, loading saved metrics")
            seed_metrics.append(_read_json(seed_metrics_path))
            continue

        if emb_model is None:
            emb_model, tokenizer = embeddings.load_embedding_and_tokenizer(emb_key)
        result = cross_validation(
            outcome_space=outcome_space,
            data_splits=cv_splits[seed_idx],
            tokenizer=tokenizer,
            emb_model=emb_model,
            enc_type=enc_name,
            enc_kwargs=enc_kwargs,
            regressor_kwargs=dict(config.REGRESSOR_KWARGS),
            opt_kwargs=dict(config.OPT_KWARGS),
            device=device,
            save_to=save_to,
        )
        seed_metrics.append(result["metrics"])
        with open(seed_metrics_path, "w", encoding="utf-8") as f:
            json.dump(result["metrics"], f, indent=2)

    summary = summarize_metrics(seed_metrics)
    summary["run_id"] = run["id"]
    summary["enc"] = enc_name
    summary["emb"] = emb_key
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[{run['id']}] acc {summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f} | "
        f"macro_f1 {summary['macro_f1_mean']:.4f}±{summary['macro_f1_std']:.4f}"
    )
    return summary


def run_enabled(run_ids=None, device=None, force=False):
    """Run all enabled RUNS, or only those whose id is in run_ids."""
    if run_ids is None:
        runs = config.enabled_runs()
    else:
        known_ids = {run["id"] for run in config.RUNS}
        unknown = [run_id for run_id in run_ids if run_id not in known_ids]
        if unknown:
            raise KeyError(f"Unknown run ids: {unknown}")
        runs = [config.run_by_id(run_id) for run_id in dict.fromkeys(run_ids)]

    outcome_space, cv_splits, devtest_splits = load_paws_cv_and_devtest()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Persist devtest index lists for analysis (texts are reloaded from PAWS).
    devtest_meta = []
    for seed, split in zip(config.SEEDS, devtest_splits):
        devtest_meta.append(
            {
                "seed": seed,
                "devtest_ids": [int(i) for i in split["devtest_ids"]],
                "test_ids": [int(i) for i in split["test_ids"]],
            }
        )
    with open(config.RESULTS_DIR / "devtest_splits.json", "w", encoding="utf-8") as f:
        json.dump(devtest_meta, f, indent=2)

    summaries = []
    for run in runs:
        summaries.append(run_one(run, outcome_space, cv_splits, device=device, force=force))
    return summaries, devtest_splits


def main(argv=None):
    parser = argparse.ArgumentParser(description="PAWS pair CV experiments")
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        default=None,
        help="Run only this run id (repeatable). Default: all enabled.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device string (default: config.DEVICE)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain runs that already have saved results.",
    )
    args = parser.parse_args(argv)
    device = torch.device(args.device) if args.device else config.DEVICE
    run_enabled(run_ids=args.run_ids, device=device, force=args.force)


if __name__ == "__main__":
    main()
