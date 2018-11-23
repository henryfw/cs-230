import numpy as np
import matplotlib.pyplot as plt
import threading
import glob
import random
import scipy.misc

from audio import spec2wav, wav2spec, read_wav, write_wav



def process_files(files, thread_id):
    sr = 22050
    n_fft = 512
    win_length = 400
    hop_length = 80
    duration = 2  # sec

    counter = 0

    data_x = []
    data_x_reduced_2 = []
    data_x_reduced_3 = []

    for file in files:
        counter += 1
        print("Reading: " + file + " " + str(counter) + " " + str(thread_id))

        wav_x = read_wav(file, sr, duration)
        spec_x, _ = wav2spec(wav_x, n_fft, win_length, hop_length, False)

        spec_x_reshaped = spec_x.reshape((spec_x.shape[0], spec_x.shape[1], 1))

        r2 = scipy.misc.imresize(spec_x, (int(spec_x.shape[0]/2), int(spec_x.shape[1]/2)), 'bilinear')
        r2 = r2.reshape((r2.shape[0], r2.shape[1], 1))
        r3 = scipy.misc.imresize(spec_x, (int(spec_x.shape[0]/3), int(spec_x.shape[1]/3)), 'nearest')
        r3 = r3.reshape((r3.shape[0], r3.shape[1], 1))

        data_x.append(spec_x_reshaped)
        data_x_reduced_2.append(r2)
        data_x_reduced_3.append(r3)

    np.save("/Volumes/USB/cs230/spec_{}_{}sec/data_x".format(thread_id, duration), data_x)
    np.save("/Volumes/USB/cs230/spec_{}_{}sec/data_x_reduced_2".format(thread_id, duration), data_x_reduced_2)
    np.save("/Volumes/USB/cs230/spec_{}_{}sec/data_x_reduced_3".format(thread_id, duration), data_x_reduced_3)


if __name__ == '__main__':

    for i in range(1, 7):
        files = glob.glob("H:/cs230/wav_{}/*".format(i))

        process_files(files, i)


    # converted_wav = spec2wav(spec, n_fft, win_length, hop_length, 600)
    # write_wav(converted_wav, sr, 'a.wav')
    # plt.pcolormesh(spec)
    # plt.ylabel('Frequency')
    # plt.xlabel('Time')
    # plt.savefig("a.png")

    print("Done!")


