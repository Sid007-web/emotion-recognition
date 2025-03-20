import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import pickle

model_path = "C:\\Users\\Siddharth\\Desktop\\Programs\\mlp_model"
with open(model_path, "rb") as f:
    mlp = pickle.load(f)


def extract(file_name, mfcc, chroma, mel):
    with sf.SoundFile(file_name) as sound_file:
        x = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(x))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=x, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

def predict(file_path):
    feature = extract(file_path, chroma=True, mfcc=True, mel=True)
    feature = feature.reshape(1, -1)
    emotion = mlp.predict(feature)[0]
    return emotion

st.title("Emotion Recognition from Speech")
st.write("Upload a .wav file to predict the emotion.")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    emotion = predict("temp.wav")
    st.write(f"Predicted Emotion: **{emotion}**")