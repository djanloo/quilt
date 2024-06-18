import numpy as np
import matplotlib.pyplot as plt

def bin_spikes(spikes, points_per_bin = 10):
    binned_signal = np.sum( spikes[:(len(spikes)//points_per_bin)*points_per_bin].reshape(-1, points_per_bin),
                        axis=1).squeeze()
    return binned_signal

o1, o2 = np.loadtxt("osc_history.txt", unpack=True)
sp = np.loadtxt("spiking_history.txt")

plt.plot(o1*1000)
plt.plot(o2*1000)
plt.plot(bin_spikes(sp))
plt.show()