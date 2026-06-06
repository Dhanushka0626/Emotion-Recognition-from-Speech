import json
import pickle
import tempfile

import librosa
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model


MODEL_PATH = "speech_emotion_cnn_final.keras"
LABEL_ENCODER_PATH = "label_encoder.pkl"
CONFIG_PATH = "config.json"
PLAYLISTS_PATH = "playlists.json"


@st.cache_resource
def load_saved_files():
    model = load_model(MODEL_PATH)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    with open(PLAYLISTS_PATH, "r") as f:
        playlists = json.load(f)

    return model, label_encoder, config, playlists


def pad_or_truncate(spec, max_len):
    if spec.shape[1] < max_len:
        pad_width = max_len - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="constant")
    else:
        spec = spec[:, :max_len]

    return spec


def preprocess_audio(file_path, config):
    signal, sr = librosa.load(
        file_path,
        sr=config["sample_rate"],
        duration=config["duration"],
        offset=config["offset"],
    )

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=config["n_mels"],
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = pad_or_truncate(mel_db, config["max_len"])

    mel_db = (mel_db + 80) / 80
    mel_db = mel_db[np.newaxis, ..., np.newaxis]

    return mel_db


st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎧",
    layout="centered",
)

st.title("🎧 Speech Emotion Recognition")
st.write(
    "Upload a speech audio file. The app predicts the emotional tone "
    "and recommends playlists based on the result."
)

try:
    model, label_encoder, config, playlists = load_saved_files()
except Exception as e:
    st.error("Error loading model or required files.")
    st.exception(e)
    st.stop()


uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "ogg"],
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    try:
        file_suffix = "." + uploaded_file.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        features = preprocess_audio(temp_path, config)

        prediction = model.predict(features, verbose=0)[0]
        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        emotion = label_encoder.inverse_transform([predicted_index])[0]

        st.subheader("Prediction Result")
        st.success(f"Detected emotional tone: **{emotion.capitalize()}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")

        st.subheader("Recommended Playlists")

        recommended = playlists.get(emotion, [])

        if recommended:
            for playlist in recommended:
                st.markdown(f"🎵 [{playlist['name']}]({playlist['url']})")
        else:
            st.info("No playlists found for this emotion.")

        with st.expander("Show prediction probabilities"):
            for label, prob in zip(label_encoder.classes_, prediction):
                st.write(f"{label}: {prob * 100:.2f}%")

    except Exception as e:
        st.error("Something went wrong while processing the audio file.")
        st.exception(e)
