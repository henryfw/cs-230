import keras


import numpy as np
import matplotlib.pylab as plt


from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, Dropout


def BidLstm(maxlen, max_features):
    inp = Input(shape=(maxlen, max_features, ))
    x = Bidirectional(LSTM(max_features, return_sequences=True, dropout=0.25,
                           recurrent_dropout=0.25))(inp)
    x = Attention(maxlen)(x)
    model = Model(inputs=inp, outputs=x)

    return model


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim





batch_size = 512
epochs = 50


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


model = BidLstm(x_train.shape[1], x_train.shape[2])

model.compile(loss='mse',
              optimizer='adam',
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