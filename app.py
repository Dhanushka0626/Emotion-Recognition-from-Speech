import streamlit as st
import numpy as np
import librosa
import json
import pickle
from tensorflow.keras.models import load_model

# Load saved files
model = load_model("speech_emotion_cnn_final.keras")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("config.json", "r") as f:
    config = json.load(f)
    
with open("playlists.json", "r") as f:
    playlists = json.load(f)

st.title("Speech Emotion Recognition")
st.write("Upload a speech audio file and the model will predict the emotional tone.")

uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "ogg"]
)

def pad_or_truncate(spec, max_len):
    if spec.shape[1] < max_len:
        pad_width = max_len - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="constant")
    else:
        spec = spec[:, :max_len]
    return spec

def preprocess_audio(file_path):
    signal, sr = librosa.load(
        file_path,
        sr=config["sample_rate"],
        duration=config["duration"],
        offset=config["offset"]
    )

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=config["n_mels"]
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = pad_or_truncate(mel_db, config["max_len"])

    mel_db = (mel_db + 80) / 80
    mel_db = mel_db[np.newaxis, ..., np.newaxis]

    return mel_db

if uploaded_file is not None:
    st.audio(uploaded_file)

    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    features = preprocess_audio("temp_audio.wav")

    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)

    emotion = label_encoder.inverse_transform([predicted_index])[0]

    st.subheader("Prediction Result")
    st.write(f"Detected emotional tone: **{emotion}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
    
    st.subheader("Recommended Playlists")

    recommended = playlists.get(emotion, [])

    for playlist in recommended:
        st.markdown(f"🎵 [{playlist['name']}]({playlist['url']})")