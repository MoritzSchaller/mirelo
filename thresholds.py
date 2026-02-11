"""
This module computes the logit thresholds for 'speech' and 'music' labels. 
A small subset of AudioSet is used as a reference.
"""

from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import load_dataset
from tqdm import tqdm

from filter_pipeline.classify import AudioClassifier
from filter_pipeline.audio_io import load_audio_av


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
    
    df['audio'] = df['audio'].str['bytes']
    df = df.drop(columns=["labels"])
    
    classifier = AudioClassifier(["speech", "music"])
    sr = classifier.sr

    batch_size = 64
    batch_indices = range(0, len(df), batch_size)
    n_batches = len(df) // batch_size

    with Pool(cpu_count()) as p:

        processed_batches: list[pd.DataFrame] = []
        for i in tqdm(batch_indices, total=n_batches):

            batch = df.iloc[i:i + batch_size] 
            audios = p.map(load_audio_av, batch["audio"])

            batch_results = classifier.classify_batch(audios)
            batch_results_df = pd.DataFrame(batch_results)

            combined_df = pd.concat([batch.reset_index(drop=True).drop(columns=["audio"]), batch_results_df], axis=1)
            processed_batches.append(combined_df)


    classified_df = pd.concat(processed_batches, ignore_index=True)
    classified_df.to_parquet(logits_file)

    print(classified_df)


def compute_thresholds(logits_file: Path) -> dict[str, float]:
    """Find decision boundaries."""

    df = pd.read_parquet(logits_file)
    
    df["has_speech"] = df["human_labels"].apply(lambda x: "Speech" in x)
    df["has_music"] = df["human_labels"].apply(lambda x: "Music" in x)

    _plot_histograms(df)


def _plot_histograms(df: pd.DataFrame):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(data=df, x='speech', hue='has_speech', bins=50, alpha=0.6, ax=axes[0])
    axes[0].axvline(0, color='green', linestyle='--', linewidth=2)

    sns.histplot(data=df, x='music', hue='has_music', bins=50, alpha=0.6, ax=axes[1])
    axes[1].axvline(0, color='green', linestyle='--', linewidth=2)

    plt.tight_layout()
    plt.show()
    
    


def main():
    n_samples = 2000
    validation_file = Path(f"data/validation_{n_samples}.parquet")
    if not validation_file.exists():
        print("Validation data file not found. Downloading.")
        download_validation_dataset(validation_file, 2000)
    
    logits_file = Path(f"data/validation_{n_samples}_with_logits.parquet")
    if not logits_file.exists():
        print("Computing classification logits")
        compute_classification_logits(validation_file, logits_file)
    
    compute_thresholds(logits_file)


if __name__ == "__main__":
    main()
    
    