import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import rtmixer
import math
import sounddevice as sd

# Parameters
FRAMES_PER_BUFFER = 512
NFFT = 4096
NOVERLAP = 128
window = NFFT * 6
downsample = 10
channels = 1

# Globals for data storage and device info
plotdata = None
ringBuffer = None
image = None

def create_specgram(frame):
    global plotdata
    
    spec, freqs, t = plt.mlab.specgram(plotdata[:,-1], Fs=samplerate)
    xmin, xmax = np.min(t) - pad_xextent, np.max(t) + pad_xextent
    extent = xmin, xmax, freqs[0], freqs[-1]
    epsilon = 1e-10
    arr = np.flipud(10. * np.log10(spec + epsilon))

    return arr, extent

def update_plot(frame):
    global plotdata

    while ringBuffer.read_available >= FRAMES_PER_BUFFER:
        read, buf1, buf2 = ringBuffer.get_read_buffers(FRAMES_PER_BUFFER)
        assert read == FRAMES_PER_BUFFER
        buffer = np.frombuffer(buf1, dtype='float32')
        buffer.shape = -1, channels
        buffer = buffer[::downsample]

        assert buffer.base.base == buf1
        shift = len(buffer)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = buffer
        ringBuffer.advance_read_index(FRAMES_PER_BUFFER)

    arr, _  = create_specgram(frame)
    image.set_array(arr)
    image.set_clim(vmin=np.min(arr), vmax=np.max(arr))  # Dynamically adjust color scale
    return image,

def start_audio_stream():
    global plotdata, ringBuffer, image, samplerate, pad_xextent

    device_info = sd.query_devices(device=1, kind='input')
    samplerate = device_info['default_samplerate']

    pad_xextent = (NFFT - NOVERLAP) / samplerate / 2
    length = int(window * samplerate / (1000 * downsample))
    plotdata = np.zeros((length, channels))

    stream = rtmixer.Recorder(device=None, channels=channels, blocksize=FRAMES_PER_BUFFER,
                              latency='low', samplerate=samplerate)
    ringbufferSize = 2**int(math.log2(3 * samplerate))

    ringBuffer = rtmixer.RingBuffer(channels * stream.samplesize, ringbufferSize)

    fig, ax = plt.subplots(figsize=(10, 5))
    arr, extent = create_specgram(0)
    image = plt.imshow(arr, animated=True, extent=extent)
    fig.colorbar(image)

    ani = FuncAnimation(fig, update_plot, interval=1, blit=True, cache_frame_data=False)
    
    with stream:
        ringBuffer = rtmixer.RingBuffer(channels * stream.samplesize, ringbufferSize)
        action = stream.record_ringbuffer(ringBuffer)
        plt.show()

if __name__ == "__main__":
    start_audio_stream()