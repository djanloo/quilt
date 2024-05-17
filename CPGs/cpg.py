from quilt.builder import SpikingNetwork
import numpy as np 
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, freqz, welch
from scipy.integrate import simpson
from scipy import stats

burn_in_millis = 10
sampling_frequency = 1e3

def bin_spikes(spikes, points_per_bin = 10):
    binned_signal = np.sum( spikes[:(len(spikes)//points_per_bin)*points_per_bin].reshape(-1, points_per_bin),
                        axis=1).squeeze()
    return binned_signal

def beta_power(sn, population):    
    spikes = sn.populations[population].get_data('spikes')
    binned_spikes = bin_spikes(spikes)[burn_in_millis:]
    T = len(binned_spikes)
    f, PSD = welch(binned_spikes, 
                   sampling_frequency, 
                   nperseg = T/2, # Takes at least 3 windows
                   noverlap= T/4,
                   nfft=None, 
                   scaling='density', 
                   window='hamming')

    beta_mask = (f>=12)&(f<30)
    low_gamma_mask = (f>=30)&(f<50)
    high_gamma_mask = (f>=50)&(f<90)
    
    norm_beta = simpson(PSD[beta_mask], x=f[beta_mask])
    fmean = simpson(f[beta_mask]*PSD[beta_mask]/norm_beta, x=f[beta_mask])

    spectral_norm = simpson(PSD, x=f)
    result = dict(
        fmax = f[np.argmax(PSD)],
        fmean=fmean,
        
        norm_beta_power=simpson(PSD[beta_mask], x=f[beta_mask])/spectral_norm,
        norm_low_gamma_power=simpson(PSD[low_gamma_mask], x=f[low_gamma_mask])/spectral_norm,
        norm_high_gamma_power=simpson(PSD[high_gamma_mask], x=f[high_gamma_mask])/spectral_norm,

        beta_power = simpson(PSD[beta_mask], x=f[beta_mask]),
        low_gamma_power = simpson(PSD[low_gamma_mask], x=f[low_gamma_mask]),
        high_gamma_power = simpson(PSD[high_gamma_mask], x=f[high_gamma_mask]),

        entropy=stats.entropy(PSD/spectral_norm),
    )
    
    return result

def firing_rate(sn, population):
    binned_spikes = bin_spikes(sn.populations[population].get_data('spikes'))
    instantaneous_fr = binned_spikes/sn.populations[pop].n_neurons*1000
    return np.mean(instantaneous_fr)

def bandpass(data, edges, sample_rate, poles = 5):
    sos = butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data

def get_PSD(sn, pop, frequency_resolution=0.5, smooth=False):
    spikes = sn.populations[pop].get_data('spikes')
    binned_spikes = bin_spikes(spikes)
    binned_spikes = binned_spikes[burn_in_millis:]
    T = len(binned_spikes)
    # print(f"Mean firing rate {pop} is {np.sum(spikes)/sn.populations[pop].n_neurons/(Tlong/1000) :.1f} Hz")

    N = sampling_frequency/frequency_resolution
    
    if N > T/2:
        print(N)
        print(f"Not enough points to achieve resolution of {frequency_resolution}")
        N = T/2

    nfft = 10_000 if smooth else None
    f, PSD = welch(binned_spikes, 
                   sampling_frequency, 
                   nperseg=N, 
                   noverlap=N/2,
                   nfft=nfft,
                   scaling='density', 
                   window='hamming')
    # print(f"F resolution { f[1] - f[0] :.2f} Hz")
    norm = simpson(PSD, x=f)
    return f, PSD
#####################################################################
net = SpikingNetwork.from_yaml("CPGs/basic_net.yaml", "CPGs/neurons.yaml")

net.monitorize_spikes()
net.monitorize_states()

net.build()
net.run(time=500)



plt.plot( *get_PSD(net, "A"))
plt.show()
# plt.plot(net.populations['A'].get_data("spikes"))
# plt.show()

