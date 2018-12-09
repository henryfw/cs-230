import numpy as np
import matplotlib.pyplot as plt


def save_spec(spec, file, invert=False):
    if invert:
        spec = np.swapaxes(spec, 0, 1)

    plt.pcolormesh(spec)
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.savefig(file)
    plt.clf()



data1 = np.load("H:/data/data1_with_dup.npy")
data2 = np.load("H:/data/data2_with_dup.npy")


spec1 = data1[0]
spec2 = data2[0]

save_spec(spec1, "images/1.png", True)
save_spec(spec2, "images/2.png", True)