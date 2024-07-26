import gradio as gr
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import time

def save_audio(audio):
    samplerate, data = audio  # Unpack sample rate and audio data
    data = np.array(data).astype(np.float32)  # Ensure the audio data is a numpy array and in float32 format
    data /= np.max(np.abs(data))  # Normalize the audio data
    
    return spectrogram(samplerate, data)

def spectrogram(Fs, audiodata):
    f, tt, Sxx = signal.spectrogram(audiodata, Fs)

    plt.pcolormesh(tt, f, Sxx, shading='gouraud')
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig('spectrogram.png')
    plt.close()

    return 'spectrogram.png'

def audio_time_update(audio):
    while True:
        spectrogram = save_audio(audio)
        yield spectrogram
        time.sleep(1)


interface = gr.Interface(
    fn=save_audio, 
    inputs=gr.Audio(sources="microphone", type="filepath"), 
    outputs=gr.Image(type="filepath"),
    title="Audio Recorder"
)

if __name__ == "__main__":
    interface.launch(share=True)