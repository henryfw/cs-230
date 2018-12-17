import numpy as np
import time
from Attention import Attention
import keras
from keras.layers import Dense, Flatten
from keras.layers import GRU, LSTM, Bidirectional, TimeDistributed, CuDNNGRU, CuDNNLSTM, Bidirectional, Lambda, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Sequential
import matplotlib.pylab as plt
import sys
import scipy
from keras.callbacks import TensorBoard

from audio import spec2wav, wav2spec, read_wav, write_wav

import keras.backend as K
K.set_image_data_format('channels_last')


batch_size = 300
epochs = 20 #2000
test_size_fraction = 0.02 # per batch


img_x, img_y =  300, 257
input_shape = (img_x, img_y)




input1 = Input(shape=input_shape)
x1 = LSTM(units=256, return_sequences=True, dropout=0.4, activation='tanh' )(input1)
input2 = concatenate([x1, input1], axis=2)
x2 = LSTM(units=256, return_sequences=True, dropout=0.4, activation='tanh' )(input2)
input3 = concatenate([x2, x1, input1], axis=2)
x3 = LSTM(units=256, return_sequences=True, dropout=0.5, activation='tanh' )(input3)
input4 = concatenate([x3, x2, x1, input1], axis=2)
d1 = TimeDistributed(Dense(512, activation = 'tanh'))(input4)
d1 = TimeDistributed(Dropout(0.4))(d1)
d2 = TimeDistributed(Dense(256, activation = 'tanh'))(d1)
d2 = TimeDistributed(Dropout(0.4))(d2)
out = TimeDistributed(Dense(img_y, activation = 'sigmoid'))(d2)

model = keras.models.Model(inputs=input1, outputs=out)


model.compile(loss='mse',
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])


ts = time.time()

current_part = 0

def get_current_part():
    return current_part


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        if batch % 10 == 1:
            model.save("models/run_{}_{}_{}.h5".format(ts, get_current_part(), batch))
            self.acc.append(logs.get('acc'))

history = AccuracyHistory()

tensorboard = TensorBoard(log_dir="logs/{}".format(ts))


number_of_parts = 2

for i in range(number_of_parts + 1):
    part = np.load("H:/data/data1_{}.npy".format(i))

    data1 = np.load("H:/data/data1_with_dup_{}.npy".format(i))
    data2 = np.load("H:/data/data2_with_dup_{}.npy".format(i))

    test_size = len(data1)

    x_train = data1[:-test_size]
    x_test = data1[-test_size:]

    y_train = data2[:-test_size]
    y_test = data2[-test_size:]


    print('x_train shape:', x_train.shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[history, tensorboard])

    current_part += 1
