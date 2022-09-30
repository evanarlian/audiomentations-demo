import itertools
from typing import Tuple
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import audiomentations as A

from helper import helper_classes


def aug_sidebar() -> list:
    """
    Create list of choices, adding a new one if full.
    Shows every params for current selected transforms.
    """
    # streamlit works by caching the previous input (cache by key) and
    # the while loop simply recreates the list everytime, but since
    # previous inputs are cached, it will break at fresh value
    incr = itertools.count()
    add_new = "➕ Add new"
    helpers = [add_new] + sorted(helper_classes.keys())
    selected = []
    aug_helpers = []
    while True:
        # this sidebar has implicit key by incrementing the label string
        sel = st.sidebar.selectbox(f"Audiomentations no. {len(selected)+1}", helpers)
        if sel == add_new:
            break
        selected.append(sel)
        # aug needs explicit key because it can contain many widget
        ah = helper_classes[sel](keygen=incr)
        ah.render()
        aug_helpers.append(ah)
    return aug_helpers


def info_sidebar() -> None:
    """Show app info"""
    # TODO
    pass


def audio_input() -> Tuple[np.ndarray, int, str]:
    """
    Performs audio loading related tasks.
    Returns tuple of (audio_data, sample_rate, filename)
    """
    # audio samples key value pair
    audio_samples = {path.name: path for path in Path("audio_samples/").iterdir()}

    # audio input settings
    selected_samples = st.selectbox("Use sample audio...", sorted(audio_samples.keys()))
    uploaded_file = st.file_uploader("...or upload you own.")
    st.caption("*All audio inputs will be converted to mono.*")
    resample = st.radio(
        "Resample to", [16000, 22050, 44100, "Use original"], index=0, horizontal=True
    )

    # handle if user is using custom or sample song
    if uploaded_file is not None:
        # librosa cannot read .mp3 BytesIO but can read .mp3 files
        # we need to fake create audio files with the matching suffix
        audio_name = uploaded_file.name
        temp = NamedTemporaryFile(suffix=Path(audio_name).suffix)
        temp.write(uploaded_file.getvalue())
        temp.seek(0)
        audio_path = temp.name
    else:
        audio_name = selected_samples
        audio_path = audio_samples[selected_samples]

    # read audio file (either fake or real)
    try:
        audio_arr, sr = librosa.load(
            audio_path, sr=None if resample == "Use original" else resample
        )
    except Exception:
        st.error(f"Error opening '{audio_name}'.")
        st.stop()
    finally:
        if uploaded_file is not None:
            temp.close()  # close (and auto delete) tempfile even if error occurs

    return audio_arr, sr, audio_name


def visualize_wave(audio_arr: np.ndarray, sr: int, audio_name: str) -> None:
    """Shows wav plot, mel spectogram, and audio player."""

    # plot wave on top and mel spec on the bottom
    fig, axs = plt.subplots(2, sharex=True, constrained_layout=True)
    librosa.display.waveshow(audio_arr, sr=sr, ax=axs[0])  # TODO gap on `barking.wav`
    axs[0].set_ylabel("Amplitude")
    mel_spec = librosa.feature.melspectrogram(y=audio_arr, sr=sr)
    db_mel_spec = librosa.amplitude_to_db(mel_spec)
    librosa.display.specshow(db_mel_spec, sr=sr, ax=axs[1], x_axis="time", y_axis="mel")
    fig.suptitle(audio_name)
    st.pyplot(fig)

    # we need to get the bytes for streamlit audio
    # and soundfile cannot save to a BytesIO so we make a fake file
    with NamedTemporaryFile(suffix=".wav") as temp:
        soundfile.write(temp, audio_arr, sr)
        temp.seek(0)
        audio_bytes = temp.read()
    st.audio(audio_bytes)


def main():

    st.set_page_config(layout="wide")
    st.title("🎹 Audiomentations Demo")
    aug_helpers = aug_sidebar()

    input_col, metadata_col = st.columns(2, gap="large")
    with input_col:
        st.header("Select audio")
        audio_before, sr, audio_name = audio_input()
    with metadata_col:
        # TODO metadata df might be good
        st.header("Input metadata")
        st.write(sr)
        st.write(audio_before.shape[0] / sr)

    # augment the audio
    aug = A.Compose([ah.aug_instance for ah in aug_helpers])
    audio_after = aug(audio_before, sample_rate=sr)

    before_col, after_col = st.columns(2, gap="large")
    with before_col:
        st.header("Before")
        visualize_wave(audio_before, sr, audio_name)
    with after_col:
        st.header("After")
        visualize_wave(audio_after, sr, audio_name)

    # TODO docs so we can understand augs + copy paste


if __name__ == "__main__":
    main()
