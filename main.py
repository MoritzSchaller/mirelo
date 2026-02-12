"""This module is the entrypoint for the entire data loading and filtering pipeline."""



import pandas as pd


from filter_pipeline.vggsound import get_vggsound_dataset
from filter_pipeline.classify import AudioClassifier, classify_multiprocessed



def main():
    
    print("Loading VGGSound dataset ...")
    df = get_vggsound_dataset(n_shards=1)
    df = df.head()
    print("Classifying ...")
    
    label_scores_df = classify_multiprocessed(df["path"])
    print(label_scores_df)
    df = pd.concat((df, label_scores_df), axis=1)
    print(df)
    
    print("Filtering ...")
    speech_threshold = 0.5 # based on histogram
    music_threshold = 0.5
    df = df[(df["speech_score"] < speech_threshold) & (df["music_score"] < music_threshold)]

    print("Saving ...")
    df.to_json("result_filtered.jsonl", orient="records", lines=True)
    
    print("Done!")




if __name__ == "__main__":
    main()