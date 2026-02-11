from pathlib import Path

import pytest
import numpy as np

from filter_pipeline import audio_io

media_path = Path(__file__).parent / "media"
video_paths = tuple(media_path.glob("*"))
sample_rates = [16000, 22050, 44100, 48000, 96000]


@pytest.mark.parametrize("path", video_paths)
@pytest.mark.parametrize("sr", sample_rates)
def test_av(path, sr):
    y = audio_io.load_audio_av(path, sr)

    assert isinstance(y, np.ndarray)
    assert len(y.shape) == 1, "Should be one channel only"
    assert y.shape[0] > 0, "Should contain frames"


@pytest.mark.parametrize("path", video_paths)
@pytest.mark.parametrize("sr", sample_rates)
def tests_ffmpeg(path, sr):
    y = audio_io.load_audio_ffmpeg(path, sr)

    assert isinstance(y, np.ndarray)
    assert len(y.shape) == 1, "Should be one channel only"
    assert y.shape[0] > 0, "Should contain frames"