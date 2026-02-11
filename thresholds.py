"""
This module computes the logit thresholds for 'speech' and 'music' labels. 
The AudioSet Dataset is used as a reference.
"""


from pathlib import Path

from datasets import load_dataset

import pandas as pd



def download_validation_dataset(parquet_file: Path, n_samples: int):
    ds = load_dataset(
        "agkphysics/AudioSet", "balanced",
        split="test",
        streaming=True
    )

    small_ds = ds.take(n_samples)
    small_ds.to_parquet(parquet_file)


def compute_thresholds(parquet_file: Path):
    pass


if __name__ == "__main__":

    validation_file = Path("data/validation_1000.parquet")

    if not validation_file.exists():
        print("Validation data file not found. Downloading.")
        download_validation_dataset(validation_file, 1000)
    
    compute_thresholds(validation_file)
    