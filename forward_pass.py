import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from audio import spec2wav, wav2spec, read_wav, write_wav


if __name__ == '__main__':


    sr = 22050
    n_fft = 512
    win_length = 400
    hop_length = 80
    duration = 2 # sec

    # plt.savefig("a-output.png")
    data_x = np.load("G:/cs230/np_2sec/data_x.npy")
    data_y = np.load("G:/cs230/np_2sec/data_y2.npy")

    x_test = np.load("C:/cs230/np_2sec/x_test.npy")
    y_test = np.load("C:/cs230/np_2sec/y_test.npy")
    #
    # model = load_model('models/run_1_28.h5')
    #
    # prediction = model.predict( x_test[0:1,:,:,:] )
    # prediction = prediction[0]


    # converted_wav = spec2wav(spec, n_fft, win_length, hop_length, 600)

    # write_wav(converted_wav, sr, 'a.wav')


    plt.pcolormesh(data_x[0,:,:])
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.savefig("a-input.png")

    #
    # plt.pcolormesh(prediction[:,:,0])
    # plt.ylabel('Frequency')
    # plt.xlabel('Time')


    plt.pcolormesh(data_y[0,:,:])
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.savefig("a-label.png")


