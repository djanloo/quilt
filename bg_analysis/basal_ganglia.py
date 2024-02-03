#!/usr/bin/env python
# coding: utf-8

# # Dopamine effects on BG
# 
# **Results**: increase in beta-range spectral power is observed without dopamine modulation of CTX input to MSN-D1 and MSN-D2. Also, I forgot modulation on GPeTI. This means that something is unnecessary to the onset of beta oscillations.
# 
# **Method**: I used the network model from Ortone and the model of dopamine modulation from Lindahl(2016). In particular, if an attribute $x$ of the network is subject to dopaminergic modulation, the effect of dopamine levels are modeled by
# 
# $$ x(\alpha) = x_0(1+\chi_x (\alpha -\alpha_0))$$
# where $\alpha$ represents the dopamine level and $x_0$ is the value of the parameter at standard dopamine ($\alpha_0 = 0.8$, Lindahl). In the following, $\chi_x$ will be called generalized susceptibility.

# In[10]:


import yaml
from rich import print

NEURONS_FILE = "lindahl_neurons.yaml"
            # print(f"initialized parameter {param_name} with value {self.params_value[param_name]} "+
                #   f"and range {self.params_range[param_name]}")ons.yaml"
NETWORK_FILE = "lindahl_network.yaml"
DOPAMINE_FILE = "lindahl_dopamine_susceptibility.yaml"

with open(DOPAMINE_FILE, "r") as dopfile:
    print(yaml.safe_load(dopfile))


# In[11]:


import numpy as np
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2
# get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib ipympl

# FSN [10-20] Hz 
# D1, D2 [0.5–2.5] Hz
# GPe-TI [40–60] Hz
# GPe-TA [5–15] Hz 
# and STN [12–20] Hz


# ## Parameters of the simulation

# In[12]:


Tlong = 1000  # ms
dt = 0.1      # ms

points_per_bin = 1/dt     # bins 1 ms wide
sampling_frequency = 1e3  # 1 kHz sampling frequency
burn_in_millis = 600      # the first part of the record to discard (ms)

# Params for trials
n_trials = 10
Tshort = 1500 # ms


# ## Utils & plots

# In[13]:


from scipy.signal import butter, sosfiltfilt, freqz, welch
from scipy.integrate import simpson
from scipy.stats import entropy

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
    print(f"f peak {f[np.argmax(PSD)]}")
    beta_mask = (f>12)&(f<30)
    return np.trapz(PSD[beta_mask], x=f[beta_mask])

def bandpass(data, edges, sample_rate, poles = 5):
    sos = butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data


# In[14]:


def plot_spectrum(sn, scale="log"):
    fig, axes = plt.subplots(len(sn.populations), 1, sharex=True)
    for ax, pop in zip(axes, sn.populations):
        spikes = sn.populations[pop].get_data('spikes')
        binned_spikes = bin_spikes(spikes)
        binned_spikes = binned_spikes[burn_in_millis:]
        T = len(binned_spikes)
        print(f"Mean firing rate {pop} is {np.sum(spikes)/sn.populations[pop].n_neurons/(Tlong/1000) :.1f} Hz")
        f, PSD = welch(binned_spikes, 
                       sampling_frequency, 
                       nperseg=T/2, 
                       noverlap=T/4,
                       nfft=None, 
                       scaling='density', 
                       window='hamming')
    
        norm = simpson(PSD, x=f)
        ax.plot(f, PSD/norm, label=pop)
        
        ax.set_ylabel(pop)
        ax.set_yscale(scale)
    print(f"F resolution { f[1] - f[0] :.2f} Hz")
    # plt.yscale('log')
    # plt.legend()
    # plt.ylabel("normalized PSD")
    plt.xlabel("Frequency [Hz]")
    plt.xlim(0,150)
    fig.set_figheight(10)
    
def plot_signals(sn):
    fig, axes = plt.subplots(len(sn.populations), 1, sharex=True)
    for ax, pop in zip(axes, sn.populations):
        binned_spikes = bin_spikes(sn.populations[pop].get_data('spikes'))
        instantaneous_fr = binned_spikes/sn.populations[pop].n_neurons*1000
        
        tt = np.linspace(0, len(instantaneous_fr)/1000, len(instantaneous_fr))
        ax.plot(tt, instantaneous_fr)
        ax.plot(tt, bandpass(instantaneous_fr, [12, 24], sampling_frequency), label="beta")
        ax.plot(tt, bandpass(instantaneous_fr, [30, 140], sampling_frequency), label="gamma")
        ax.set_ylabel(pop)
    # ax.set_xlim(1.5, 2)
    fig.suptitle("Instantaneous firing rate [Hz]")
    ax.set_xlabel("time [s]")
    fig.set_figheight(10)


# ## Building the network

# In[15]:


from quilt.interface.spiking import set_verbosity
set_verbosity(1)


# In[16]:


from quilt.builder import NeuronCatalogue, ParametricSpikingNetwork

neuron_catalogue = NeuronCatalogue.from_yaml(NEURONS_FILE)
sn = ParametricSpikingNetwork.from_yaml(NETWORK_FILE, 
                                        DOPAMINE_FILE, 
                                        neuron_catalogue)
sn.monitorize_spikes()


# In[17]:


# from quilt.view import plot_graph
# fig, ax = plt.subplots()
# plot_graph(sn)
# plt.show()


# ## Control case (Healthy subject)

# In[ ]:


sn.set_parameters(dopamine=0.8, striatum_scale=-0.85, non_striatum_scale=-0.6)
sn.build(progress_bar=True)
print(sn.features_dict)
sn.run(dt=dt, time=Tlong)

# In[10]:


plot_spectrum(sn, scale="linear")


# In[11]:


plot_signals(sn)
plt.gca().set_xlim(3.5, 4)


# ## Lesioned case (PD)

# In[15]:


sn.set_parameters(dopamine=0.1, striatum_scale=-0.85, non_striatum_scale=-0.6)
sn.build(progress_bar=False)
sn.run(dt=dt, time=Tlong)


# In[16]:


plot_spectrum(sn, scale="linear")


# In[22]:


plot_signals(sn)
plt.gca().set_xlim(2.5, 3)

plt.show()
# In[ ]:




