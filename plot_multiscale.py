import numpy as np
import matplotlib.pyplot as plt

dT = 1.0
dt = 0.1
delay = 10

def bin_spikes(spikes, points_per_bin = 10):
    binned_signal = np.sum( spikes[:(len(spikes)//points_per_bin)*points_per_bin].reshape(-1, points_per_bin),
                        axis=1).squeeze()
    return binned_signal

osc = np.loadtxt("osc_history.txt")
sp = np.loadtxt("spiking_history.txt")

if len(osc.shape) == 1:
    osc= osc[None, :]


time = np.arange(len(osc[0]))*dT
for i, o in enumerate(osc):
    plt.plot(time, o, label = f"osc_{i}", marker="o")

# plt.step(np.arange(0, len(bin_spikes(sp))) , bin_spikes(sp), label="binned_spikes")


def rate_of_spikes(spikes, time_bins, N_samples):

    # Time in ms

    N_bins = len(time_bins) - 1
    rate = time_bins*0
    for i in range(N_bins):
        mask = (spikes > time_bins[i])&(spikes <= time_bins[i+1])
        rate[i] = np.sum(mask)/(time_bins[i+1] - time_bins[i])/N_samples

    return rate*1000

n, tsp = np.loadtxt("test_inh_poiss.txt", unpack=True)
N = np.max(n)
time_bins = np.linspace(0, len(sp)*dt, 200)
rates = rate_of_spikes(tsp, time_bins, N)

# plt.step(time_bins, rates, where="mid", label="Transducer spikes")


osc_interp = np.loadtxt("osc_interpol.txt")
plt.plot(np.arange(len(osc_interp))*dt, osc_interp, label = "oscillator past NCE interpolated")


t, rate = np.loadtxt("td_incoming_rates.txt", unpack=True)
plt.scatter(t - delay, rate, label="TD incoming rate (shifted)", s=2, color="k", zorder=22)



plt.xlim(13, 17)
plt.ylim(0.266, 0.276)
plt.legend()
plt.show()