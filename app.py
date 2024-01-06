import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from io import BytesIO
import matplotlib.pyplot as plt
import time  # Ensure this import is here

# Assuming 'commands' contains your class labels
commands = np.array(['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'])
MODEL_PATH = 'save_model'

# Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Define the spectrogram function
def get_spectrogram(waveform, sample_rate=16000, max_length=16000):
    # Ensure waveform is a float32
    waveform = tf.cast(waveform, dtype=tf.float32)

    # If waveform is shorter than max_length, pad with zeros
    if tf.shape(waveform)[0] < max_length:
        padding = max_length - tf.shape(waveform)[0]
        zero_padding = tf.zeros([padding], dtype=tf.float32)
        waveform = tf.concat([waveform, zero_padding], 0)
    # If waveform is longer, truncate it to the max_length
    else:
        waveform = waveform[:max_length]

    # Calculate the STFT
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    
    # We take the magnitude of the STFT output to get the spectrogram
    spectrogram = tf.abs(spectrogram)
    
    # Add a channel dimension, where the 'channel' is just the real part of the spectrogram
    spectrogram = spectrogram[..., tf.newaxis]
    
    return spectrogram

# Streamlit user interface
st.title('Command Capture Classification')
st.write("Team 22 - Design of a Command Capture Method for Virtual Reality Training Simulation System")

st.markdown('<div style="background-color: #608397; padding: 10px; color: white;">'
            'Note: Upload a WAV audio file (up to 200MB) to classify commands. Just drag and drop your file into the area above.'
            '</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a WAV file to classify.", type=["wav"])
if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    audio, sr = librosa.load(BytesIO(bytes_data), sr=16000)
    
    # Get the spectrogram
    spectrogram = get_spectrogram(audio)
    
    # Make batch dimension
    spectrogram = spectrogram[tf.newaxis, ...]
    
    # Start the timer
    start_time = time.time()
    
    # Predict the class
    predictions = model(spectrogram, training=False)
    
    # End the timer
    end_time = time.time()
    
    # Calculate the prediction time
    prediction_time = end_time - start_time
    
    predicted_index = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_class = commands[predicted_index]
    
    # Display prediction
    st.write(f'The model predicts: {predicted_class}')
    
    # Display prediction time as a bar graph
    plt.figure(figsize=(5, 3))
    plt.bar(['Prediction Time'], [prediction_time], color='skyblue')
    plt.ylabel('Time (seconds)')
    plt.title('Time Taken for Prediction')
    st.pyplot(plt)
