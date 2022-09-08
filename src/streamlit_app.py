from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st



def input_form() -> Tuple[np.ndarray, int, str]:
    """
    Performs audio loading related tasks.
    Returns tuple of (audio_data, sample_rate, filename)
    """
    # audio samples key value pair
    audio_samples = {path.name: path for path in Path("audio_samples/").iterdir()}

    # use form
    with st.form("input_form"):
        st.header("Select audio")
        selected_samples = st.selectbox("Use sample audio...", audio_samples.keys())
        uploaded_file = st.file_uploader("...or upload you own.")
        st.caption("*All audio inputs will be converted to mono.*")
        resample = st.checkbox("Resample to 22050 Hz", value=True)
        submitted = st.form_submit_button("Augment")

    # stop rendering if not submitted
    if not submitted:
        st.stop()

    # load audio (will be converted to mono by default)
    try:
        audio_arr, sr = librosa.load(
            uploaded_file
            if uploaded_file is not None
            else audio_samples[selected_samples],
            sr=22050 if resample else None,
        )
    except Exception as e:
        st.error(e)
        st.stop()

    audio_name = uploaded_file.name if uploaded_file is not None else selected_samples
    return audio_arr, sr, audio_name


def visualize_wave(audio_arr: np.ndarray, sr: int, audio_name: str) -> None:

    # plot wave on top and mel spec on the bottom
    fig, axs = plt.subplots(2, constrained_layout=True)
    librosa.display.waveshow(audio_arr, sr=sr, ax=axs[0])
    axs[0].set_ylabel("Amplitude")
    mel_spec = librosa.feature.melspectrogram(y=audio_arr, sr=sr)
    db_mel_spec = librosa.amplitude_to_db(mel_spec)
    librosa.display.specshow(db_mel_spec, ax=axs[1], x_axis="time", y_axis="mel")
    fig.suptitle(audio_name)
    st.pyplot(fig)

    # saving to a fake file for streamlit audio
    with NamedTemporaryFile(suffix=".wav") as temp:
        soundfile.write(temp, audio_arr, sr)
        with open(temp.name, "rb") as f:
            audio_bytes = f.read()
    st.audio(audio_bytes)



def main():
    st.title("🎹 Audiomentations Demo")

    # TODO choose aug here

    audio_arr, sr, audio_name = input_form()
    
    st.header("Before")
    visualize_wave(audio_arr, sr, audio_name)

    # TODO do aug here

    st.header("After")
    visualize_wave(audio_arr, sr, audio_name)


    # TODO docs so we can understand augs + copy paste


if __name__ == "__main__":
    main()
