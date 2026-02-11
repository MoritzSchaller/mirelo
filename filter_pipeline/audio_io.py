import subprocess

from pathlib import Path

import numpy as np
import av


def load_audio_ffmpeg(path: Path, sr: int=48000) -> np.ndarray[np.float32]:
    """
    Extract audio from video file as numpy array using ffmpeg. 
    ffmpeg library has to be installed manually.
    """

    command = [
        'ffmpeg',
        '-i', str(path),
        '-f', 'f32le',
        '-acodec', 'pcm_f32le',
        '-ac', '1',
        '-ar', str(sr),
        '-'
    ]

    result = subprocess.run(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        raise Exception(f"Unable to load audio: {result.stderr}")
    
    audio = np.frombuffer(result.stdout, np.float32)
        
    return audio



def load_audio_av(path: Path, sr: int=48000) -> np.ndarray:
    """
    Extract audio from video file as numpy array using pyav. 
    This is faster for many small files because it does not spawn a new process for each extracton.
    """

    resampler = av.AudioResampler(format="fltp", layout="mono", rate=sr)
    
    with av.open(str(path)) as container:
        frames = [
            out_frame.to_ndarray()[0]
            for packet in container.demux(audio=0)
            for frame in packet.decode()
            for out_frame in resampler.resample(frame)
        ] + [out.to_ndarray()[0] for out in resampler.resample(None)]

    return np.concatenate(frames, dtype=np.float32)
