import gradio as gr # gradio UI 활용
import numpy as np # audio를 ndarray로 받아 처리함
import matplotlib.pyplot as plt # spectrogram 출력 및 저장
import torch
import torchaudio

#torchaudio.set_audio_backend("sox_io")

def save_audio_new(stream, audio):
    """
    input : stream(이전 loop의 audio data 묶음), audio(현재 loop의 audio data)
    output : sr(sample rate), stream(stream+audio)
    
    audio를 sample rate(int)와 data(ndarray)로 분리시킨 뒤 data를 Tensor로 변화시킨다.
    stream이 None일 때(최초 loop) stream=data(Tensor형태)
    stream이 None이 아닐 때(2회 이상 loop) stream 뒤에 data를 붙인다.
    이후 torchaudio를 이용해 stream을 wav file로 저장
    """
    sr, y = audio
    data_np = y.astype(np.float32)
    data_ts = torch.tensor(data_np)
    
    if stream is not None : 
        stream = torch.cat((stream, data_ts))
    else:
        stream = data_ts
    
    torchaudio.save('inputaudiolive.wav', src=stream, sample_rate=sr)
    return sr, stream

def get_spec_new():
    """
    input:none(문제 해결 시 (filepath, filename)으로 변경)
    output:spectrogram

    이전에 저장한 wav file을 불러와 waveform과 sample rate로 분리
    이를 이용해 spectrogram 출력
    """
    waveform, sample_rate = torchaudio.load('inputaudiolive.wav')
    transform = torchaudio.transforms.Spectrogram()
    spec = transform(waveform, Fs=sample_rate)
    return spec

def save_plot_new(spec):
    """
    input:spectrogram(문제 해결 시 filepath와 filename 추가)
    output:None

    입력으로 받은 spectrogram을 plotting 후 주어진 파일명에 따라 저장
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(spec.log2()[0, :, :].detach().numpy(), cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Frames')
    plt.ylabel('Frequency')
    plt.savefig('spectrogramlive.png', format='png')
    plt.close()

def main_new(stream, audio):
    """
    input : stream, audio
    output : stream, spectrogram
    """
    save_audio_new(stream, audio)
    spec = get_spec_new()
    save_plot_new(spec)
    return stream, 'spectrogramlive.png'

interface = gr.Interface(
    fn=main_new, 
    inputs=["state", gr.Audio(sources="microphone", streaming=True)],
    outputs=["state", gr.Image(type="filepath")],
    title="Audio Plotter",
    live=True
)

if __name__ == "__main__":
    interface.launch()
