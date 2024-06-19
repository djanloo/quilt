import numpy as np
import matplotlib.pyplot as plt

def rate_of_spikes(spikes, time_bins, N_samples):

    # Time in ms

    N_bins = len(time_bins) - 1
    rate = time_bins*0
    for i in range(N_bins):
        mask = (spikes > time_bins[i])&(spikes <= time_bins[i+1])
        rate[i] = np.sum(mask)/(time_bins[i+1] - time_bins[i])/N_samples

    return rate*1000

n, tsp = np.loadtxt("test_inh_poiss.txt", unpack=True)

time_bins = np.linspace(0, 2000, 100)
rates = rate_of_spikes(tsp, time_bins, 500)

plt.step(time_bins, rates, where="mid")

plt.show()