import numpy as np
import matplotlib.pyplot as plt
import threading
import glob
import random
import scipy.misc

from audio import spec2wav, wav2spec, read_wav, write_wav



def process_files(files, duration = 2):
    sr = 22050
    n_fft = 512
    win_length = 400
    hop_length = 80

    counter = 0

    data_x = []
    data_y = []

    for file in files:
        counter += 1
        if counter % 1000 == 0: print("Counter: " + str(counter) + " on folder ")

        try:
            wav_x = read_wav(file, sr, duration)
            spec_x, _ = wav2spec(wav_x, n_fft, win_length, hop_length, False)
            data_x.append(np.swapaxes(spec_x, 0, 1))

            wav_y = read_wav(file.replace('wav_1', 'wav_6'), sr, duration)
            spec_y, _ = wav2spec(wav_y, n_fft, win_length, hop_length, False)
            data_y.append(np.swapaxes(spec_y, 0, 1))

        except:
            print("error doing processing: " + file)

    np.save("H:/cs230/spec_{}sec/data_x".format(duration), data_x)
    np.save("H:/cs230/spec_{}sec/data_y".format(duration), data_y)


if __name__ == '__main__':

    files = glob.glob("H:/cs230/wav_1/*")

    for duration in [1,2,3]:
        process_files(files, duration)


    # converted_wav = spec2wav(spec, n_fft, win_length, hop_length, 600)
    # write_wav(converted_wav, sr, 'a.wav')
    # plt.pcolormesh(spec)
    # plt.ylabel('Frequency')
    # plt.xlabel('Time')
    # plt.savefig("a.png")

    print("Done!")


