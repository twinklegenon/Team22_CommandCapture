import streamlit as st
import tensorflow as tf
import numpy as np
from io import BytesIO
import librosa

# Assuming 'commands' contains your class labels as in your notebook
commands = np.array(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'])
# Replace with the actual path to your saved model
MODEL_PATH = 'save_model'

# Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Define the spectrogram function
def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

# Streamlit user interface
st.title('Audio Command Classification')
st.write('Upload a WAV file to classify.')

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])
if uploaded_file is not None:
    # Read audio file
    bytes_data = uploaded_file.read()
    audio, sr = librosa.load(BytesIO(bytes_data), sr=16000)
    
    # Get the spectrogram
    spectrogram = get_spectrogram(audio)
    
    # Make batch dimension
    spectrogram = spectrogram[tf.newaxis, ...]
    
    # Predict the class
    predictions = model(spectrogram, training=False)
    predicted_index = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_class = commands[predicted_index]
    
    # Display prediction
    st.write(f'The model predicts: {predicted_class}')
