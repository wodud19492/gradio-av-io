import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read
from scipy import signal

audio_name = 'inputaudio.wav'
save_path = '/home/data/jylee/gradio-av-io/'

def save_audio(audio):
    sr, y = audio
    data_np =np.array(y).astype(np.float32)
    data_np /= np.max(np.abs(data_np))
    #data_ts = torch.tensor(data)
    write(save_path + audio_name, sr, data_np)
    #torchaudio.save(audio_name, src=data_ts, sample_rate=samplerate)

def get_specgram(audio_name):
    """
    입력: 
    audio file path

    출력
    spectrogram(np.ndarray)
    """
    data, samplerate = read(save_path+audio_name)
    f, t, Sxx = signal.spectrogram(data, samplerate)
    return f, t, Sxx
    #waveform, sample_rate = torchaudio.load(audio_name, normalize=True)
    #transform = torchaudio.transforms.Spectrogram()
    #spectrogram = transform(waveform, Fs=sample_rate)


def save_plot(f, t, Sxx, save_path):
    """
    spectrogram 입력
    출력: None
    """
    
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.savefig(save_path + 'spectrogram.png')
    plt.close()
    

def main(audio):
    """
    입력
    audio(str) :wav 파일의 경로
    
    출력
    image_path(str) : 출력 스펙트로그램의 경로

    """
    save_audio(audio)
    f, t, Sxx = get_specgram(audio_name)
    save_plot(f, t, Sxx, save_path)
    return save_path + 'spectrogram.png'


interface = gr.Interface(
    fn=main, 
    inputs=gr.Audio(type="numpy", label="Record your audio"), 
    outputs=gr.Image(type="filepath"),
    title="Audio Plotter"
)

if __name__ == "__main__":
    interface.launch(share=True)