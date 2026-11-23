"""Write the devtest analysis frame of every trained run to exports/analysis_df.csv.

Run from the repo root: uv run python part2/project/exports/export_analysis.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from part2.project import analysis, config, embeddings, experiment

EXPORT_DIR = Path(__file__).resolve().parent
TOKENIZER_EMB_KEY = "fasttext300"


def add_sentences(analysis_df: pd.DataFrame, devtest_splits) -> pd.DataFrame:
    """Add sentence1/sentence2 columns, matching rows to devtest pairs by position within each checkpoint."""
    sentences_by_run = {
        run: [pair for pair, _label in split["devtest"]]
        for run, split in enumerate(devtest_splits)
    }
    row_position = analysis_df.groupby(["system", "run", "fold"]).cumcount()
    pairs = [sentences_by_run[run][position] for run, position in zip(analysis_df["run"], row_position)]
    analysis_df["sentence1"] = [sentence1 for sentence1, _sentence2 in pairs]
    analysis_df["sentence2"] = [sentence2 for _sentence1, sentence2 in pairs]
    return analysis_df


def add_run_descriptors(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Add the embedding and encoder of each system as emb/enc columns."""
    analysis_df["emb"] = analysis_df["system"].map(lambda system: config.run_by_id(system)["emb"])
    analysis_df["enc"] = analysis_df["system"].map(lambda system: config.run_by_id(system)["enc"])
    return analysis_df


def build_export_frame(systems) -> pd.DataFrame:
    _outcome_space, _cv_splits, devtest_splits = experiment.load_paws_cv_and_devtest()
    _emb_model, tokenizer = embeddings.load_embedding_and_tokenizer(TOKENIZER_EMB_KEY)

    system_predictions = analysis.collect_devtest_predictions(systems, devtest_splits)
    analysis_df = analysis.build_analysis_frame(devtest_splits, system_predictions, tokenizer)
    analysis_df = add_sentences(analysis_df, devtest_splits)
    return add_run_descriptors(analysis_df)


def main():
    systems = analysis.trained_run_ids()
    print("systems:", systems, flush=True)
    analysis_df = build_export_frame(systems)

    output_path = EXPORT_DIR / "analysis_df.csv"
    analysis_df.to_csv(output_path, index=False)
    print(f"wrote {len(analysis_df)} rows to {output_path}")
    print(analysis_df.groupby("system")["correct"].agg(["mean", "size"]))


if __name__ == "__main__":
    main()
