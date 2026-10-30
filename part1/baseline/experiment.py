import json
import numpy as np
from pathlib import Path
from collections import OrderedDict
from tqdm.auto import tqdm

from . import smlp
from . import clslib


ENCODER_TYPES = {
    'indicator': clslib.TokenIndicator,
    'bow': clslib.BagOfWords,
}


def system_name(model_option: str, encoder_option: str, pair_mode: str, tokenizer_name: str):
    """
    Name of a system, used as its subfolder under the results folder.

    model_option: 'linear' or 'mlp'
    encoder_option: e.g. 'indicator' or 'bow'
    pair_mode: one of clslib.PAIR_MODES
    tokenizer_name: e.g. the output of tokenization.bpe_name
    """
    return f"{model_option}_{encoder_option}-enc_{pair_mode}_{tokenizer_name}"


def make_pair_encoder(tokenizer, encoder_option: str, pair_mode: str):
    """
    Build a PairEncoder around a TokenIndicator ('indicator') or BagOfWords ('bow') encoder for the given tokenizer.
    """
    if encoder_option not in ENCODER_TYPES:
        raise ValueError(f"Unknown encoder_option={encoder_option}, choose from {tuple(ENCODER_TYPES)}")
    return clslib.PairEncoder(ENCODER_TYPES[encoder_option](tokenizer), pair_mode)


def check_saved_settings(path, settings: dict):
    """
    Save settings to the json file at path, or, if that file exists, make sure the saved settings 
    are the same as the current ones (so that results saved with other settings are never reused).

    path: json file next to the saved results (e.g. {system_folder}/system_config.json)
    settings: dict with all settings that affect the saved results
    """
    path = Path(path)
    settings = json.loads(json.dumps(settings))  # same types as after loading from disk
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if saved != settings:
            raise ValueError(
                f"{path} was saved with different settings:\nsaved: {saved}\ncurrent: {settings}\n"
                "Delete the saved results to recompute them, or change the settings back."
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)


def cross_validation_experiment(data_splits, outcome_space, encoder: clslib.Encoder, vectorizer, model_option: str, save_to="tmp", 
                                model_args=dict(), device="cpu"):
    """
    Trains and evaluates k models using a k-fold data split for cross validation. 
    Folds that were already trained (that have a {save_to}/fold-{fold}/fold_result.json) are loaded instead of retrained.
    This function returns a dictionary with 
        - models: the paths to the k models trained (the model objects aren't returned, they are stored in disk)
        - predictions: the predictions by each of the k models
        - accuracies: the accuracy of each trained model on its corresponding validation partition
    
    data_splits: the training/validation folds
    outcome_space: NamedClasses for the dataset
    encoder: an instance of Encoder
    vectorizer: function (inputs, encoder) -> SciPy CSR matrix (e.g. smlp.vectorize)
    model_option: 'linear' or 'mlp'
    save_to: folder where to store models
    model_args: dict with training arguments for smlp.train_neural_network_model
    device: where to train
    """

    models, predictions, accuracies = OrderedDict(), OrderedDict(), OrderedDict()
    
    for fold in tqdm(data_splits.keys()):
        training_set = data_splits[fold]["training"]
        validation_set = data_splits[fold]["validation"]

        fold_folder = Path(f"{save_to}/fold-{fold}")
        fold_folder.mkdir(parents=True, exist_ok=True) 
        result_path = fold_folder / "fold_result.json"

        if not result_path.exists():
            metrics, best_epoch, fold_predictions = smlp.train_neural_network_model(
                training_set, validation_set, len(outcome_space), encoder, vectorizer, save_to=f'{fold_folder}/model', 
                model_type=model_option, device=device, **model_args
            )
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump({
                    'accuracy': metrics[best_epoch]['val_metrics']['accuracy'],
                    'best_epoch': best_epoch,
                    'predictions': [int(p) for p in fold_predictions],
                    'metrics': metrics,
                }, f, indent=2)

        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        models[fold] = f'{fold_folder}/model'
        predictions[fold] = np.array(result['predictions'])
        accuracies[fold] = result['accuracy']

    return {'models': models, 'predictions': predictions, 'accuracies': accuracies}


def run_system(name, cross_validation_sets, seeds: list, outcome_space, encoder, vectorizer, model_option: str, 
               model_args: dict, system_config: dict, results_folder, device="cpu"):
    """
    Run cross_validation_experiment once per seed for one system. Return a list with one result per run (seed).

    name: name of the system (its subfolder under the results folder)
    cross_validation_sets: list with the folds of each run (one entry per seed, same order as seeds)
    seeds: a list of seeds (they name subfolders under the system folder)
    outcome_space: NamedClasses for the dataset
    encoder: an instance of Encoder
    vectorizer: function (inputs, encoder) -> SciPy CSR matrix
    model_option: 'linear' or 'mlp'
    model_args: dict with training arguments for smlp.train_neural_network_model
    system_config: settings of the system, checked against previously saved results (see check_saved_settings)
    results_folder: where everything is saved
    device: where to train
    """
    check_saved_settings(f"{results_folder}/{name}/system_config.json", system_config)
    runs = []
    for run, seed in enumerate(seeds):
        runs.append(
            cross_validation_experiment(
                data_splits=cross_validation_sets[run],
                outcome_space=outcome_space,
                encoder=encoder,
                vectorizer=vectorizer,
                model_option=model_option,
                save_to=f"{results_folder}/{name}/seed-{seed}",
                model_args=model_args,
                device=device,
            )
        )
    return runs


def predict_labels(model_path, x_sparse, device="cpu"):
    """
    Load a trained model and predict (via the most probable class decision rule) the labels of all rows of x_sparse.

    model_path: path prefix of the model (see smlp.load)
    x_sparse: SciPy CSR matrix with one encoded input per row
    device: where to run the model
    """
    model = smlp.load(model_path, device)
    return smlp.predict_probs(x_sparse, model).argmax(axis=1)


def single_and_ensemble_predictions(system_runs: list, x_sparse, device="cpu", seed=0):
    """
    Return the predictions of a single model (first seed, first fold, as in the tutorial) and of the majority vote 
    ensemble of all models (all seeds and folds) of a system.

    system_runs: output of run_system
    x_sparse: SciPy CSR matrix with one encoded input per row
    device: where to run the models
    seed: for breaking ties in the majority vote
    """
    all_predictions = []
    for run_data in system_runs:
        for fold, model_path in run_data['models'].items():
            all_predictions.append(predict_labels(model_path, x_sparse, device))
    all_predictions = np.array(all_predictions)  # [num_models, num_inputs]
    return {
        'single': all_predictions[0],
        'ensemble': clslib.majority_vote(all_predictions, seed=seed),
    }
