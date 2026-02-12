### Mirelo Technical Test


Using the VGGSound dataset (https://huggingface.co/datasets/Loie/VGGSound/tree/main)


## Task

We would like you to design and execute a filtering pipeline to identify videos in the VGGSound dataset that contain only sound effects.

* In this context, sound effects are defined as all sounds that are not music and not speech.
* For each selected video, we would like a corresponding text description that describes the audio content.
* The final output should be a JSON Lines file named sfx_filtered.jsonl.

Expected Minimal Output Format:

{ "video_id": "id_to_video1", "audio_text_description": "Sound of a bird singing" } 
{ "video_id": "id_to_video2", "audio_text_description": "Car engine and road noise" } 
{ "video_id": "id_to_video3", "audio_text_description": "..." } 
... 
{ "video_id": "id_to_videoN", "audio_text_description": "..." }

# Additional Considerations

* Please take efficiency considerations into account (e.g., scalability, runtime, data handling) and any design trade-offs you make. In the end such pipelines need to run on millions of files.
* Describe how you would check the quality of your solution (both in terms of data quality and performance).
* You are free to use any tools, libraries, models, or frameworks you consider appropriate (open-source or otherwise). Please briefly justify your choices.

There is no single “correct” solution. We are primarily interested in your reasoning, design decisions, and clarity of thought.
Technical Setup

* Please set up an environment where you can run your prepared homework live (e.g. on your local machine using the CPU, or a simple Google Colab).
* If you need access to a dedicated GPU please let us know so we can provide you with the resources.
    For the interview, please turn off AI assistance.



## Solution

1) Manually inspect the dataset on huggingface
    * VGGSound does not adhere to modern huggingface dataset format
        * has to be manually downloaded and indexed
        * filter_pipeline.vggsound acts as an adapter module

2) Lots of mp4 files to extract
    * compared two different ways of decoding
        * ffmpeg is a good default choice, but requires spawning a process per call, has to be installed manually
        * av calls ffmpegs c libraries directly and can be installed including those libraries in the wheels 
    * benchmark (using multiprocessing) reveals that av is much faster

3) Audio classificaton networks for speech and music
    * Research models on Huggingface
        * CLAP: zero shot capability, great for extending to different labels later, promising
        * BEATs: better accuracy, big model
        * YamNet: tiny model, ideal for CPU
    * Verification Dataset with speech and music labels necessary
        * AudioSet fits the bill
        * Use subset of test split only
        * Find thresholds for speech and music scores from score historgrams (see thresholds.py) 
    * Try CLAP model 
        * Audio and text embeddings with cosine similarity between them
        * Labels don't change -> precompute text embeddings on init
        * Label separation ends up beeing really bad (see histogram_clap.svg)
    * Try YamNet
        * Small model can be loaded to each CPU worker process
        * Label separation looks much better (see histogram_yamnet.png)
    * Decision: YamNet

4) Design considerations
    * Should run on a local CPU for easy development, cloud can come later
    * Use multiprocessing for speed
    * Use parquet files for efficient data storage
    * Keep it simple: no frameworks like Dask, Ray, Prefect, Airflow that take time to set up properly.
    * Audio CLAP and YamNet could be easily GPU accelerated later
        * enable CUDA
        * optimize batch size 
        * possibly load the model to each GPU multiple times for best saturation
    * Advanced scenario for later
        * Ray framework allows distributed compute and storage with great GPU resource management
        * Prefect framework for workflow orchestration and better observability

5) How to check quality
    * Run on subset of data to measure execution time
    * Manually listen to filtered results (both included and excluded items)
    * Check accuracy on labeled datasets