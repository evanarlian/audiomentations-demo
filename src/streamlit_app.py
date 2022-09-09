from tempfile import NamedTemporaryFile, SpooledTemporaryFile
from pathlib import Path
from typing import Tuple, final

import numpy as np
import soundfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st


def aug_sidebar():
    pass


def input_form() -> Tuple[np.ndarray, int, str]:
    """
    Performs audio loading related tasks.
    Returns tuple of (audio_data, sample_rate, filename)
    """
    # audio samples key value pair
    audio_samples = {path.name: path for path in Path("audio_samples/").iterdir()}

    # use form
    with st.form("input_form"):
        selected_samples = st.selectbox("Use sample audio...", audio_samples.keys())
        uploaded_file = st.file_uploader("...or upload you own.")
        st.caption("*All audio inputs will be converted to mono.*")
        resample = st.checkbox("Resample to 22050 Hz", value=True)
        submitted = st.form_submit_button("Augment")

    # stop rendering if not submitted
    if not submitted:
        st.stop()

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
        audio_arr, sr = librosa.load(audio_path, sr=22050 if resample else None)
    except Exception:
        st.error(f"Error opening '{audio_name}'.")
        st.stop()
    finally:
        if uploaded_file is not None:
            temp.close()  # close (and auto delete) tempfile even if error occurs

    return audio_arr, sr, audio_name


def visualize_wave(audio_arr: np.ndarray, sr: int, audio_name: str) -> None:

    # plot wave on top and mel spec on the bottom
    fig, axs = plt.subplots(2, sharex=True, constrained_layout=True)
    librosa.display.waveshow(audio_arr, sr=sr, ax=axs[0])
    axs[0].set_ylabel("Amplitude")
    mel_spec = librosa.feature.melspectrogram(y=audio_arr, sr=sr)
    db_mel_spec = librosa.amplitude_to_db(mel_spec)
    librosa.display.specshow(db_mel_spec, ax=axs[1], x_axis="time", y_axis="mel")
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

    # TODO choose aug here
    input_col, metadata_col = st.columns(2, gap="large")

    with input_col:
        st.header("Select audio")
        audio_arr, sr, audio_name = input_form()

    with metadata_col:
        # TODO metadata
        st.header("Input metadata")
        st.write(audio_arr.shape[0] / sr)

    # TODO do aug here

    before_col, after_col = st.columns(2, gap="large")

    with before_col:
        st.header("Before")
        visualize_wave(audio_arr, sr, audio_name)

    with after_col:
        st.header("After")
        visualize_wave(audio_arr, sr, audio_name)

    # TODO docs so we can understand augs + copy paste


if __name__ == "__main__":
    main()
