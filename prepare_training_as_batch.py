import numpy as np
import pickle

# load all parts
number_of_parts = 22



for i in range(number_of_parts + 1):
    data1 = np.load("H:/data/data1_{}.npy".format(i))
    if len(data1) > 0:
        print("Reading part {}".format(i))

        word_counts = np.array(pickle.load( open( "H:/data/word_count_{}.pkl".format(i), "rb" ) ))

        for j in range(2, 6):
            indices = (word_counts >= j).nonzero()
            data1 = np.concatenate([data1, data1[indices]])

        randomize = np.arange(len(data1))
        np.random.shuffle(randomize)
        data1 = data1[randomize]
        np.save("H:/data/data1_with_dup_{}".format(i), data1)

        data2 = np.load("H:/data/data2_{}.npy".format(i))

        for j in range(2, 6):
            indices = (word_counts >= j ).nonzero()
            data2 = np.concatenate( [data2, data2[indices] ])


        data2 = data2[randomize]
        np.save("H:/data/data2_with_dup_{}".format(i), data2)







