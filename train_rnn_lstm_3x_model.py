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

batch_size = 1000
epochs = 100000

img_x, img_y =  50, 30


x_train = np.load("C:/cs230/np_2sec_resized/x_train.npy")/244
x_train = x_train.reshape(x_train.shape[0:3])
x_train = np.swapaxes(x_train, 1, 2)[:,0:img_x,0:img_y]
x_test = np.load("C:/cs230/np_2sec_resized/x_test.npy")/244
x_test = x_test.reshape(x_test.shape[0:3])
x_test = np.swapaxes(x_test, 1, 2)[:,0:img_x,0:img_y]

y_train = np.load("C:/cs230/np_2sec_resized/y_train.npy")/244
y_train = y_train.reshape(y_train.shape[0:3])
y_train = np.swapaxes(y_train, 1, 2)[:,0:img_x,0:img_y]
y_test = np.load("C:/cs230/np_2sec_resized/y_test.npy")/244
y_test = y_test.reshape(y_test.shape[0:3])
y_test = np.swapaxes(y_test, 1, 2)[:,0:img_x,0:img_y]



print('x_train shape:', x_train.shape)


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


model.compile(loss='mse', # categorical_crossentropy
              optimizer=keras.optimizers.Adam(lr=0.001),
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

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()