import logging
from io import BytesIO
from typing import Tuple
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import librosa
import librosa.display
from scipy.io import wavfile
import matplotlib.pyplot as plt
import audiomentations as A
from audiomentations.core.transforms_interface import BaseWaveformTransform
import streamlit as st


CLASSES = {
    # "AddBackgroundNoise": A.AddBackgroundNoise,
    "AddGaussianNoise": A.AddGaussianNoise,
    "AddGaussianSNR": A.AddGaussianSNR,
    # "AddShortNoises": A.AddShortNoises,
    # "AdjustDuration": A.AdjustDuration,
    "AirAbsorption": A.AirAbsorption,
    # "ApplyImpulseResponse": A.ApplyImpulseResponse,
    "BandPassFilter": A.BandPassFilter,
    "BandStopFilter": A.BandStopFilter,
    "Clip": A.Clip,
    "ClippingDistortion": A.ClippingDistortion,
    "Gain": A.Gain,
    "GainTransition": A.GainTransition,
    "HighPassFilter": A.HighPassFilter,
    "HighShelfFilter": A.HighShelfFilter,
    # "Lambda": A.Lambda,
    "Limiter": A.Limiter,
    "LoudnessNormalization": A.LoudnessNormalization,
    "LowPassFilter": A.LowPassFilter,
    "LowShelfFilter": A.LowShelfFilter,
    "Mp3Compression": A.Mp3Compression,
    "Normalize": A.Normalize,
    "Padding": A.Padding,
    "PeakingFilter": A.PeakingFilter,
    "PitchShift": A.PitchShift,
    "PolarityInversion": A.PolarityInversion,
    "Resample": A.Resample,
    "Reverse": A.Reverse,
    "RoomSimulator": A.RoomSimulator,
    "SevenBandParametricEQ": A.SevenBandParametricEQ,
    "Shift": A.Shift,
    "TanhDistortion": A.TanhDistortion,
    "TimeMask": A.TimeMask,
    "TimeStretch": A.TimeStretch,
    "Trim": A.Trim,
}


def show_aug_sidebar() -> list[BaseWaveformTransform]:
    """
    Create list of choices, adding a new one if full.
    Shows every params for current selected transforms.
    """
    st.sidebar.header("List of transformations")
    # streamlit works by caching the previous input (cache by key) and
    # the while loop simply recreates the list everytime, but since
    # previous inputs are cached, it will break at fresh value
    add_new = "➕ Add new"
    choices = [add_new] + sorted(CLASSES.keys())
    n_selected = 1
    selected = []
    while True:
        # this sidebar has implicit key by incrementing the label string
        sel = st.sidebar.selectbox(f"No. {n_selected}", choices)
        if sel == add_new:
            break
        n_selected += 1
        selected.append(CLASSES[sel](p=1.0))
    return selected


def show_audio_input() -> Tuple[np.ndarray, int, str]:
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
    st.markdown("**> Press *'R'* to rerun.**")

    # handle if user is using custom or sample song
    if uploaded_file is not None:
        # librosa cannot read .mp3 BytesIO but can read .mp3 files
        # we need to fake create audio files with the matching suffix
        # see https://github.com/librosa/librosa/issues/1267
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
    except Exception as e:
        err_message = f"Error opening '{audio_name}'."
        logging.exception(err_message, e)
        st.error(err_message)
        st.stop()
    finally:
        if uploaded_file is not None:
            temp.close()  # close (and auto delete) tempfile even if error occurs

    return audio_arr, sr, audio_name


def show_metadata(audio_arr: np.ndarray, sr: int, audio_name: str) -> None:
    st.json(
        {
            "filename": audio_name,
            "channel": "mono",
            "sample_rate": sr,
            "n_samples": len(audio_arr),
            "duration_in_sec": len(audio_arr) / sr,
        }
    )


def show_wave(audio_arr: np.ndarray, sr: int, audio_name: str) -> None:
    """Shows wav plot, mel spectogram, and audio player."""
    fig, axs = plt.subplots(2, constrained_layout=True)
    # plot mel first to get the bound for waveform
    mel_spec = librosa.feature.melspectrogram(y=audio_arr, sr=sr)
    db_mel_spec = librosa.amplitude_to_db(mel_spec)
    librosa.display.specshow(db_mel_spec, sr=sr, ax=axs[1], x_axis="time", y_axis="mel")
    librosa.display.waveshow(audio_arr, sr=sr, ax=axs[0], x_axis=None)
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlim(axs[1].get_xlim())
    fig.suptitle(audio_name)
    st.pyplot(fig)

    # for some reason st audio does not play np array
    # we need to get the bytes for streamlit audio
    with BytesIO() as bytesio:
        wavfile.write(bytesio, sr, audio_arr)
        st.audio(bytesio)


def show_docs(selected: list[BaseWaveformTransform]) -> None:
    """Shows the docs of selected transforms."""
    if selected == []:
        st.write("Select one transform to begin.")
    else:
        for i, s in enumerate(selected, 1):
            st.subheader(f"{i}. {s.__class__.__name__}")
            st.help(s)


def main():

    st.set_page_config(layout="wide")
    st.title("🎹 Audiomentations Demo")

    selected = show_aug_sidebar()

    input_col, metadata_col = st.columns(2, gap="large")
    with input_col:
        st.header("Select audio")
        audio_before, sr, audio_name = show_audio_input()
    with metadata_col:
        st.header("Input metadata")
        show_metadata(audio_before, sr, audio_name)

    # augment the audio
    aug = A.Compose(selected)
    audio_after = aug(audio_before, sample_rate=sr)

    before_col, after_col = st.columns(2, gap="large")
    with before_col:
        st.header("Before")
        show_wave(audio_before, sr, audio_name)
    with after_col:
        st.header("After")
        show_wave(audio_after, sr, audio_name)

    st.header("Docs")
    show_docs(selected)


if __name__ == "__main__":
    main()
