import streamlit as st
import numpy as np
import librosa
import tempfile
import joblib
from tensorflow.keras.models import load_model

# Load model and encoder
model = load_model("emotion_model.h5")
encoder = joblib.load("label_encodder.pkl")


# Feature extraction function
def extract_features(file_path):
    audio, sample_rate = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc.T, axis=0)

    return mfcc_mean


# Emoji dictionary
emoji_dict = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "neutral": "😐",
    "calm": "😌",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😲"
}


# UI
st.title("🎤 Speech Emotion Recognition System")

st.write("Upload a WAV audio file and click Submit.")

uploaded_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

if st.button("Submit"):

    if uploaded_file is None:
        st.warning("Please upload a WAV file.")

    else:

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        # Extract MFCC features
        features = extract_features(file_path)

        # Reshape for LSTM
        features = np.array(features).reshape(1, 40, 1)

        # Predict
        prediction = model.predict(features)

        predicted_class = np.argmax(prediction)

        emotion = encoder.inverse_transform([predicted_class])[0]

        confidence = np.max(prediction) * 100

        emoji = emoji_dict.get(emotion.lower(), "")

        st.success(f"Emotion Detected: {emotion.capitalize()} {emoji}")
        st.write(f"Confidence: {confidence:.2f}%")
