import numpy as np
import pickle

# load all parts
number_of_parts = 22
data1_to_concat = []
data2_to_concat = []
word_counts = []

for i in range(number_of_parts + 1):
    part = np.load("H:/data/data1_{}.npy".format(i))
    if len(part) > 0:
        print("Reading part {}".format(i))
        data1_to_concat.append(part.astype('float16'))
        part = pickle.load( open( "H:/data/word_count_{}.pkl".format(i), "rb" ) )
        word_counts = word_counts + part

data1 = np.concatenate( data1_to_concat )
data1_to_concat = None
word_counts = np.array(word_counts)

print(data1.shape)


# increase data by adding in more copies of words that are more frequent by sample from frequencies saved
for i in range(2, 6):
    indices = (word_counts >= i ).nonzero()
    data1 = np.concatenate( [data1, data1[indices] ])

print(data1.shape)


# shuffle
randomize = np.arange(len(data1))
np.random.shuffle(randomize)
data1 = data1[randomize]
np.save("H:/data/data1_with_dup_full", data1)
data1 = None


word_counts = []

for i in range(number_of_parts + 1):
    part = np.load("H:/data/data2_{}.npy".format(i))
    if len(part) > 0:
        print("Reading part {}".format(i))
        data2_to_concat.append(part.astype('float16'))
        part = pickle.load( open( "H:/data/word_count_{}.pkl".format(i), "rb" ) )
        word_counts = word_counts + part

data2 = np.concatenate( data2_to_concat )
data2_to_concat = None
word_counts = np.array(word_counts)


for i in range(2, 6):
    indices = (word_counts >= i ).nonzero()
    data2 = np.concatenate( [data2, data2[indices] ])


data2 = data2[randomize]
np.save("H:/data/data2_with_dup_full", data2)


