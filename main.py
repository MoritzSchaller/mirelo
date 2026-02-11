"""This module is the entrypoint for the entire data loading and filtering pipeline."""


from pathlib import Path
from multiprocessing import Pool, cpu_count

import pandas as pd

from tqdm import tqdm

from filter_pipeline.audio_io import load_audio_av
from filter_pipeline.vggsound import get_vggsound_dataset
from filter_pipeline.classify import HtsatAudioClassifier


# load dataset
vggsound_df = get_vggsound_dataset(n_shards=1)

# run classification
classifier = HtsatAudioClassifier(["speech", "music"])
sr = classifier.sr

batch_size = 64
batch_indices = range(0, len(vggsound_df), batch_size)
n_batches = len(vggsound_df) // batch_size

with Pool(cpu_count - 1) as p:

    processed_batches = []
    for i in tqdm(batch_indices, total=n_batches):

        batch = vggsound_df.iloc[i:i + batch_size] 
        audios = p.map(load_audio_av, batch["path"])

        batch_results = classifier.classify_batch(audios)
        batch_results_df = pd.DataFrame(batch_results)

        combined_df = pd.concat([batch.reset_index(drop=True), batch_results_df], axis=1)
        processed_batches.append(combined_df)


classified_df = pd.concat(processed_batches, ignore_index=True)
print(classified_df)

# these thresholds were computed using thresholds.py
speech_threshold = 0.5
music_threshold = 0.9

df_filtered = df[(df["speech"] < speech_threshold) & (df["music"] < music_threshold)]

# write result
df_filtered.to_json("result_filtered.jsonl", orient="records", lines=True)
