from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(data_dim, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(Dense(10, activation='softmax'))

model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data

x_train = np.random.random((1000, timesteps, data_dim))

x_train = np.load("C:/cs230/np_2sec_resized/x_train.npy")
x_train = x_train.reshape(x_train.shape[0:3])
x_train = np.swapaxes(x_train, 1, 2)
x_train = x_train[0:1000, 0:timesteps, 0:data_dim]

y_train = np.random.random((1000, timesteps, data_dim))


print(x_train.shape, y_train.shape)

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, timesteps, data_dim))

model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))
