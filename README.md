# Emotion Recognition from Speech 

This project uses a machine learning model to predict emotions from speech. It is built using Python, Streamlit, and Scikit-learn.

## Features 
- Upload a `.wav` file to predict the emotion.
- Displays the predicted emotion (e.g., Happy, Sad, Angry, etc.).
- Built with a Multi-Layer Perceptron (MLP) classifier trained on audio features like MFCCs, Chroma, and Mel Spectrogram.

## How to Run Locally 
### Steps
1. **Clone this repository**:
   ```bash
   git clone https://github.com/Sid007-web/emotion-recognition.git
2. Navigate to the project directory.
3. Install the required dependencies:
 streamlit
 librosa
 numpy
 soundfile
 scikit-learn
 pickle
4. Run the Streamlit app:
   streamlit run app.py
5. Open your browser and go to http://localhost:8501
