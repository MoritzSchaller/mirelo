"""
This module computes the logit thresholds for 'speech' and 'music' labels. 
A small subset of AudioSet is used as a reference.
"""

import os
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from datasets import load_dataset

from filter_pipeline.classify import AudioClassifier, classify_multiprocessed


def download_validation_dataset(parquet_file: Path, n_samples: int):
    """Load part of the AudioSet dataset."""

    ds = load_dataset(
        "agkphysics/AudioSet", "balanced",
        split="test",
        streaming=True
    )

    small_ds = ds.take(n_samples)
    small_ds.to_parquet(parquet_file)


def compute_classification_logits(validation_file: Path, logits_file: Path):
    """Compute the classification logits for the ClapModel"""
    
    df = pd.read_parquet(validation_file)
    
    df["audio"] = df["audio"].str["bytes"]
    df = df.drop(columns=["labels"])
    
    label_scores_df = classify_multiprocessed(df["audio"])

    df = df.drop(columns=["audio"])
    df = pd.concat((df.reset_index(), label_scores_df), axis=1)

    df.to_parquet(logits_file)

    print(df)


def compute_thresholds(logits_file: Path) -> dict[str, float]:
    """Find decision boundaries."""

    df = pd.read_parquet(logits_file)
    
    df["has_speech"] = df["human_labels"].apply(lambda x: "Speech" in x)
    df["has_music"] = df["human_labels"].apply(lambda x: "Music" in x)

    _plot_histograms(df)


def _plot_histograms(df: pd.DataFrame):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(data=df, x='speech_score', hue='has_speech', bins=50, alpha=0.6, ax=axes[0])
    axes[0].axvline(0, color='green', linestyle='--', linewidth=2)

    sns.histplot(data=df, x='music_score', hue='has_music', bins=50, alpha=0.6, ax=axes[1])
    axes[1].axvline(0, color='green', linestyle='--', linewidth=2)

    plt.tight_layout()
    plt.savefig("histogramm_yamnet.png")
    
    


def main():
    n_samples = 2000
    validation_file = Path(f"data/validation_{n_samples}.parquet")
    if not validation_file.exists():
        print("Validation data file not found. Downloading.")
        download_validation_dataset(validation_file, 2000)
    
    logits_file = Path(f"data/validation_{n_samples}_with_yamnet_scores.parquet")
    if not logits_file.exists():
        print("Computing classification logits")
        compute_classification_logits(validation_file, logits_file)
    
    compute_thresholds(logits_file)


if __name__ == "__main__":
    main()
    
    