from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

from audio import spec2wav, wav2spec, read_wav, write_wav


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, spec
    # return freqs, times, np.log(spec.T.astype(np.float32) + eps)



if __name__ == '__main__':
    # plotstft('/Users/henry/PycharmProjects/cs230/text/raw/140_8.wav')

    sr = 22050
    n_fft = 512
    win_length = 400
    hop_length = 80
    duration = 2 # sec

    wav = read_wav( 'raw/140_8.wav', sr, duration )
    spec, _ = wav2spec(wav, n_fft, win_length, hop_length, False)

    converted_wav = spec2wav(spec, n_fft, win_length, hop_length, 600)

    write_wav(converted_wav, sr, 'a.wav')


    plt.pcolormesh(spec)
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.savefig("a.png")


