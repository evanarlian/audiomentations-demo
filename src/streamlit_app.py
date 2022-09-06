from pathlib import Path

import numpy as np
import librosa
import streamlit as st


def input_form() -> tuple:
    """
    Performs audio loading related tasks.
    Returns tuple of (audio_data, sample_rate, filename)
    """
    # audio samples key value pair
    audio_samples = {path.name: path for path in Path("audio_samples/").iterdir()}

    # use form
    with st.form("input_form"):
        st.header("Select audio")
        st.caption("*All audio inputs will be converted to mono.*")
        selected_samples = st.selectbox("Use sample audio...", audio_samples.keys())
        uploaded_file = st.file_uploader("...or upload you own")
        resample = st.checkbox("Resample to 22050 Hz", value=True)
        submitted = st.form_submit_button("Augment")

    # stop rendering if not submitted
    if not submitted:
        st.stop()

    # load audio (will be converted to mono by default)
    try:
        audio_arr, sr = librosa.load(
            uploaded_file if uploaded_file is not None else audio_samples[selected_samples],
            sr=22050 if resample else None,
        )
    except Exception as e:
        st.error(e)
        st.stop()

    audio_name = uploaded_file.name if uploaded_file is not None else selected_samples
    return audio_arr, sr, audio_name


def main():
    st.title("🎹 Audiomentations Demo")
    audio_arr, sr, audio_name = input_form()
    st.write(audio_arr.shape, sr, audio_name)


if __name__ == "__main__":
    main()
