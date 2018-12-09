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

    img_x, img_y = 50, 30

    # plt.savefig("a-output.png")
    x_test = np.load("C:/cs230/np_2sec_resized/x_test.npy")/244
    x_test = x_test.reshape(x_test.shape[0:3])
    x_test = np.swapaxes(x_test, 1, 2)[:,0:img_x,0:img_y]


    index_to_use = 0

    plt.pcolormesh(np.swapaxes(x_test[index_to_use], 0, 1))
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.savefig("a-input.png")




    y_test = np.load("C:/cs230/np_2sec_resized/y_test.npy") / 244
    y_test = y_test.reshape(y_test.shape[0:3])
    y_test = np.swapaxes(y_test, 1, 2)[:,0:img_x,0:img_y]
    plt.pcolormesh(np.swapaxes(y_test[index_to_use], 0, 1))
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.savefig("a-label.png")



    model = load_model('models/run_1544044715.6602554_3001.h5')
    prediction = model.predict( x_test[index_to_use:index_to_use+1,:,:] )
    prediction = prediction[0]
    prediction = np.nan_to_num(prediction)
    spec = np.swapaxes(prediction, 0, 1)

    plt.pcolormesh(spec)
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.savefig("a-result.png")

    # converted_wav = spec2wav(spec, n_fft, win_length, hop_length, 600)
    #
    # write_wav(converted_wav, sr, 'a.wav')
