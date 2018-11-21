import numpy as np

import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation
from keras.models import Sequential
import matplotlib.pylab as plt
import sys
import scipy
from audio import spec2wav, wav2spec, read_wav, write_wav
import keras.backend as K
K.set_image_data_format('channels_last')



batch_size = 32
epochs = 50

img_x, img_y = 128, 276


x_train = np.load("C:/cs230/np_2sec_resized/x_train.npy")
x_test = np.load("C:/cs230/np_2sec_resized/x_test.npy")

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = np.load("C:/cs230/np_2sec_resized/y_train.npy")
y_test = np.load("C:/cs230/np_2sec_resized/y_test.npy")

input_shape = (img_x, img_y, 1)

kernel_size=(5,5)

model = Sequential()
model.add(Conv2D(256, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding='same', input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Conv2D(128, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Conv2DTranspose(1, kernel_size=kernel_size, strides=(1, 1), activation='relu', border_mode='same'))

model.compile(loss='mse',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        model.save("models/run_1_{}.h5".format(batch))
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