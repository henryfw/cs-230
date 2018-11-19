from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import threading
import glob

import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

from audio import spec2wav, wav2spec, read_wav, write_wav



def process_files(files, thread_id):
    sr = 22050
    n_fft = 512
    win_length = 400
    hop_length = 80
    duration = 2  # sec

    counter = 0

    data_x = []
    data_y1 = []
    data_y2 = []

    for file in files:

        wav_x = read_wav(file, sr, duration)
        wav_y1 = read_wav(file.replace("wav_x", "wav_y1"), sr, duration)
        wav_y2 = read_wav(file.replace("wav_x", "wav_y2"), sr, duration)
        spec_x, _ = wav2spec(wav_x, n_fft, win_length, hop_length, False)
        spec_y1, _ = wav2spec(wav_y1, n_fft, win_length, hop_length, False)
        spec_y2, _ = wav2spec(wav_y2, n_fft, win_length, hop_length, False)

        data_x.append(spec_x)
        data_y1.append(spec_y1)
        data_y2.append(spec_y2)

        counter += 1
        print(file + " done in " + str(thread_id) + " " + str(counter))

        # converted_wav = spec2wav(spec, n_fft, win_length, hop_length, 600)
        # write_wav(converted_wav, sr, 'a.wav')
        # plt.pcolormesh(spec)
        # plt.ylabel('Frequency')
        # plt.xlabel('Time')
        # plt.savefig("a.png")

    np.save("/Volumes/USB/cs230/np/data_x_" + str(thread_id), data_x)
    np.save("/Volumes/USB/cs230/np/data_y1_" + str(thread_id), data_y1)
    np.save("/Volumes/USB/cs230/np/data_y2_" + str(thread_id), data_y2)

if __name__ == '__main__':


    files = glob.glob("/Volumes/USB/cs230/data_processed/*.txt")

    print("Processing " + str(len(files)) + " files:")

    files_per_part = 1000
    for i in range(0, len(files), files_per_part):
        some_files = files[i : i + files_per_part]
        process_files(some_files, i)

    # num_threads = 1
    # thread_pointers = [i for i in range(num_threads)]
    #
    # files_per_part = int( len(files)  / num_threads )
    #
    # for i in range(num_threads):
    #     some_files = files[i * files_per_part : (i+1) * files_per_part]
    #     thread_pointers[i] = threading.Thread(target=process_files, args=(some_files, i))
    #
    #
    # for i in range(num_threads):
    #     thread_pointers[i].start()
    #
    # for i in range(num_threads):
    #     thread_pointers[i].join()


    print("Done!")


