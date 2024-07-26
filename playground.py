import gradio as gr
from scipy.io.wavfile import write
import numpy as np

def save_audio(audio):
    samplerate, y = audio  # Unpack sample rate and audio data
    y = np.array(y).astype(np.float32)  # Ensure the audio data is a numpy array and in float32 format
    y /= np.max(np.abs(y))  # Normalize the audio data
    
    # Save the audio data to a WAV file
    filename = "output1.wav"
    write(filename, samplerate, y)
    return filename

interface = gr.Interface(
    fn=save_audio, 
    inputs=gr.Audio(type="numpy", label="Record your audio"), 
    outputs=gr.File(label="Download your recorded audio"),
    title="Audio Recorder",
    description="Record audio from your microphone and save it as a WAV file."
)

if __name__ == "__main__":
    interface.launch(share=True)