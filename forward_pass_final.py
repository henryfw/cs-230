import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from audio import spec2wav, wav2spec, read_wav, write_wav


def save_spec(spec, file, invert=False):
    if invert:
        spec = np.swapaxes(spec, 0, 1)

    plt.pcolormesh(spec)
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.savefig(file)
    plt.clf()



if __name__ == '__main__':


    sr = 22050
    n_fft = 512
    win_length = 400
    hop_length = 80
    duration = 2 # sec

    img_x, img_y = 150, 50

    data1 = np.load("H:/data/data1_with_dup_0.npy")
    data2 = np.load("H:/data/data2_with_dup_0.npy")


    index_to_use = 50

    x_test = data1[:index_to_use+1, :img_x, :img_y]
    y_test = data2[:index_to_use+1, :img_x, :img_y]

    model = load_model('models/run_1544633028.9742308_401.h5')
    prediction = model.predict( x_test[0:index_to_use+1,:,:] )
    prediction = np.nan_to_num(prediction)

    for i in range(index_to_use + 1):
        save_spec(x_test[i], "images/{}-input.png".format(i), True)
        save_spec(y_test[i], "images/{}-label.png".format(i), True)
        save_spec(prediction[i], "images/{}-pred.png".format(i), True)



    # converted_wav = spec2wav(spec, n_fft, win_length, hop_length, 600)
    #
    # write_wav(converted_wav, sr, 'a.wav')
