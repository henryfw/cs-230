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


batch_size = 128
epochs = 5000
test_size = 5000

skip_count = 10000

img_x, img_y = 50, 50 # original is 300, 257

# data1 = np.load("C:/cs230/data1_with_dup_full.npy")
# data2 = np.load("C:/cs230/data2_with_dup_full.npy")
data1 = np.load("H:/data/data1_with_dup_0.npy")
data2 = np.load("H:/data/data2_with_dup_0.npy")

x_train = data1[skip_count:-test_size,:img_x,:img_y]
x_test = data1[-test_size:,:img_x,:img_y]

y_train = data2[skip_count:-test_size,:img_x,:img_y]
y_test = data2[-test_size:,:img_x,:img_y]


print('x_train shape:', x_train.shape) # x_train shape: (40151, img_x, img_y)

input_shape = (img_x, img_y)

input1 = Input(shape=input_shape)
x1 = GRU(units=1024, return_sequences=True, dropout=0.4, activation='tanh' )(input1)
input2 = concatenate([x1, input1], axis=2)
x2 = GRU(units=1024, return_sequences=True, dropout=0.4, activation='tanh' )(input2)
input3 = concatenate([x2, x1, input1], axis=2)
d1 = TimeDistributed(Dense(2048, activation = 'tanh'))(input3)
d1 = TimeDistributed(Dropout(0.4))(d1)
out = TimeDistributed(Dense(img_y, activation = 'sigmoid'))(d1)

model = keras.models.Model(inputs=input1, outputs=out)


model.compile(loss='mean_absolute_error', # mean_absolute_error  mean_squared_logarithmic_error
              optimizer='rmsprop',
              metrics=['accuracy'])


ts = time.time()
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        if batch % 10 == 1:
            model.save("models/run_{}_{}.h5".format(ts, batch))
            self.acc.append(logs.get('acc'))

history = AccuracyHistory()


tensorboard = TensorBoard(log_dir="logs/{}".format(ts))


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history, tensorboard])
