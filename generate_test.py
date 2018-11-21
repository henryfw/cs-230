import numpy as np
import matplotlib.pyplot as plt

from audio import spec2wav, wav2spec, read_wav, write_wav


if __name__ == '__main__':

    sr = 22050
    n_fft = 512
    win_length = 400
    hop_length = 80
    duration = 2 # sec

    wav = read_wav( "H:\\cs230\\wav_x\\1_1.wav", sr, duration )
    spec, _ = wav2spec(wav, n_fft, win_length, hop_length, False)

    converted_wav = spec2wav(spec, n_fft, win_length, hop_length, 600)

    write_wav(converted_wav, sr, 'a.wav')


    plt.pcolormesh(spec)
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.savefig("a.png")


