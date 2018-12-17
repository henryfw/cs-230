import numpy as np
import time
from Attention import Attention
import keras
from keras.layers import Dense, Flatten
from keras.layers import GRU, LSTM, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation
from keras.models import Sequential
import matplotlib.pylab as plt
import sys
import scipy
from audio import spec2wav, wav2spec, read_wav, write_wav
import keras.backend as K
K.set_image_data_format('channels_last')

batch_size = 512
epochs = 50

img_x, img_y =  276, 128


x_train = np.load("C:/cs230/np_2sec_resized/x_train.npy")/244
x_train = x_train.reshape(x_train.shape[0:3])
x_train = np.swapaxes(x_train, 1, 2)
x_test = np.load("C:/cs230/np_2sec_resized/x_test.npy")/244
x_test = x_test.reshape(x_test.shape[0:3])
x_test = np.swapaxes(x_test, 1, 2)

print('x_train shape:', x_train.shape)

y_train = np.load("C:/cs230/np_2sec_resized/y_train.npy")/244
y_train = y_train.reshape(y_train.shape[0:3])
y_train = np.swapaxes(y_train, 1, 2)
y_test = np.load("C:/cs230/np_2sec_resized/y_test.npy")/244
y_test = y_test.reshape(y_test.shape[0:3])
y_test = np.swapaxes(y_test, 1, 2)

input_shape = (img_x, img_y)

model = Sequential()
model.add(GRU(input_shape=input_shape,  units=img_y, return_sequences=True, return_state=True, dropout=0, activation='relu'))
model.add(GRU(units=img_y, return_sequences=True, dropout=0, activation='relu'))
model.add(GRU(units=img_y, return_sequences=True, dropout=0, activation='relu'))
model.add(GRU(units=img_y, return_sequences=True, dropout=0, activation='relu'))
# model.add(LSTM(units=img_y*2, return_sequences=True, dropout=0.4, activation='relu'))
# model.add(LSTM(units=img_y, return_sequences=True, dropout=0, activation='relu'))
# model.add(Activation('sigmoid'))
# model.add(GRU(units=img_y, return_sequences=True, activation='sigmoid'))

# model.add(Bidirectional(LSTM(units=img_y, return_sequences=True, dropout=0.2), input_shape=input_shape))
# model.add(Bidirectional(LSTM(units=img_y, return_sequences=True)))
# model.add(Bidirectional(LSTM(units=img_y, return_sequences=True)))

model.compile(loss='mse',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


ts = time.time()
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        model.save("models/run_{}_{}.h5".format(ts, batch))
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()