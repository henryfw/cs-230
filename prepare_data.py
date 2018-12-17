import numpy as np
import scipy.misc
import sys
import pickle

def resize(data):
    new_data = []
    for i in range(len(data)):
        new_image = scipy.misc.imresize(data[i].reshape((256, 552)), (128, 276), 'bicubic')
        new_data.append( new_image.reshape( (128, 276, 1)) )
    return np.array(new_data)


for file in ["C:/cs230/np_2sec/x_train.npy", "C:/cs230/np_2sec/x_test.npy", "C:/cs230/np_2sec/y_train.npy", "C:/cs230/np_2sec/y_test.npy"]:
    new_file = file.replace("np_2sec", "np_2sec_resized")
    np.save(new_file, resize(np.load(file) ))



sys.exit()



data_x = np.load("G:/cs230/np_2sec/data_x.npy")
data_y = np.load("G:/cs230/np_2sec/data_y2.npy")

print(data_x.shape) # data_x.shape = (31000, 257, 552)


batch_size = 64
epochs = 50

test_count = 1000

img_x, img_y = 256, 552

x_train = data_x[:-test_count,:256,:].reshape(len(data_x)-test_count, img_x, img_y, 1)
x_test = data_x[-test_count:,:256,:].reshape(test_count, img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

np.save("C:/cs230/np_2sec/x_train", x_train)
np.save("C:/cs230/np_2sec/x_test", x_test)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


y_train = data_y[:-test_count,:256,:].reshape(len(data_x)-test_count, img_x, img_y, 1)
y_test = data_y[-test_count:,:256,:].reshape(test_count, img_x, img_y, 1)


np.save("C:/cs230/np_2sec/y_train", y_train)
np.save("C:/cs230/np_2sec/y_test", y_test)
