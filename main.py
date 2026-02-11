"""This module is the entrypoint for the entire data loading and filtering pipeline."""


from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm

from filter_pipeline.vggsound import get_vggsound_dataset
from filter_pipeline.classify import AudioClassifier



def main():
    
    print("Loading VGGSound dataset ...")
    df = get_vggsound_dataset(n_shards=1)

    print("Classifying ...")

    
    with Pool(cpu_count(), init_worker) as p:
        scores = list(tqdm(
            p.imap(process_wrapper, df["path"]),
            desc="Classifying Audio Files",
            total=len(df)
        ))

    df[["speech_score", "music_score"]] = scores

    print("Filtering ...")
    speech_threshold = 0.5
    music_threshold = 0.5
    df = df[(df["speech_score"] < speech_threshold) & (df["music_score"] < music_threshold)]

    print("Saving ...")
    df.to_json("result_filtered.jsonl", orient="records", lines=True)
    
    print("Done!")


def init_worker():
    global classifier 
    classifier = AudioClassifier()


def process_wrapper(path):
    return classifier.process_file(path)


if __name__ == "__main__":
    main()