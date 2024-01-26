# %%
# !echo $PYTHONPATH
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from memory_profiler import profile

# %load_ext autoreload
# %autoreload 2
# # %matplotlib inline
# %matplotlib ipympl

# %%
from quilt.interface.spiking import set_verbosity
set_verbosity(1)


T = 800 # simulate for 800 ms
dt = 0.1 # ms
points_per_bin = 10      # bins 1 ms wide
sampling_frequency = 1e3 # 1 kHz sampling frequency
burn_in_millis = 200     # the first part of the record to discard (ms)

# %%
from quilt.builder import NeuronCatalogue, SpikingNetwork
catalogue = NeuronCatalogue.from_yaml("tests/basal_ganglia_neurons.yaml")

sn = SpikingNetwork.from_yaml("tests/basal_ganglia_network.yaml", catalogue)

# Magic super clean params: {'poisson_rescale': 1.9739110331634744, 'weight_rescale': 1.7952043585344422}

sn.rescale_populations(1.0)
sn.rescale_connectivity(1)
sn.rescale_weights(1.79)
sn.rescale_delays(1)

poisson_rescale = 1.97

sn.build()

# %%
# from quilt.view import plot_graph
# fig, ax = plt.subplots()
# plot_graph(sn)

# %% [markdown]
# ## Input/Output

# %%
# pop_state_monitorized = "FSN"
for population in sn.populations.values():
    population.monitorize_spikes()
# sn.populations[pop_state_monitorized].monitorize_states()

# %%
poisson_inputs = dict(STN   = [500, 0.25],
                      GPeTA = [170, 0.15],
                      GPeTI = [1530, 0.25 ],
                      FSN   = [944.4, 0.5],
                      SNR   = [600, 0.55]
                    )

for pi in poisson_inputs.values():
    pi[1] *= poisson_rescale
    
for pop in poisson_inputs:
    sn.populations[pop].add_poisson_spike_injector(*poisson_inputs[pop])

sn.populations["D1"].add_poisson_spike_injector(1120, 0.45 * poisson_rescale)
sn.populations["D2"].add_poisson_spike_injector(972.927, 0.45 * poisson_rescale) 

# %% [markdown]
# ## Run and get spikes

# %%
sn.run(dt=dt, time=T)

# %%
spikes = dict()
for pop in sn.populations.keys():
    spikes[pop] = sn.populations[pop].get_data()['spikes']

# %%
def bin_spikes(spikes, points_per_bin = 10):
    binned_signal = np.sum( spikes[:(len(spikes)//points_per_bin)*points_per_bin].reshape(-1, points_per_bin),
                        axis=1).squeeze()
    return binned_signal

# %%
plt.plot(bin_spikes(spikes["D1"]))
plt.show()

# %%
del sn

# %% [markdown]
# ## Tuning
# Starts an optuna study to tune `poisson_rescale` and `weight_rescale`. As objective function the total beta-range power is chosen.

# %%
import optuna

from scipy.signal import welch

optimize_population = "GPeTA"

def build_network(poisson_rescale, rescale_weights):
    
    catalogue = NeuronCatalogue.from_yaml("tests/basal_ganglia_neurons.yaml")
    sn = SpikingNetwork.from_yaml("tests/basal_ganglia_network.yaml", catalogue)

    sn.rescale_populations(1)
    sn.rescale_connectivity(1)
    sn.rescale_weights(rescale_weights)
    sn.rescale_delays(1)
    
    sn.build(progress_bar=False)

    for pop in sn.populations.values():
      pop.monitorize_spikes()
    # sn.populations[optimize_population].monitorize_spikes()
    
    poisson_inputs = dict(STN   = [500, 0.25],
                          GPeTA = [170, 0.15],
                          GPeTI = [1530, 0.25 ],
                          FSN   = [944.4, 0.5],
                          SNR   = [600, 0.55],
                          D1    = [1120, 0.45],
                          D2    = [972.972, 0.45]
                        )
    # poisson_rescale = 0.8
    for pi in poisson_inputs.values():
        pi[1] *= poisson_rescale
        
    for pop in poisson_inputs:
        sn.populations[pop].add_poisson_spike_injector(*poisson_inputs[pop])
    
    return sn

def beta_power(sn):
    sn.run(dt=0.1, time=T)
    
    spikes = sn.populations[optimize_population].get_data()['spikes']
    binned_spikes = bin_spikes(spikes)[burn_in_millis:]
    
    f, PSD = welch(binned_spikes, 
                   sampling_frequency, 
                   nperseg = (T - burn_in_millis)/2, # Takes at least 3 windows
                   noverlap= (T - burn_in_millis)/4,
                   nfft=None, 
                   scaling='density', 
                   window='hamming')
    
    beta_mask = (f>12)&(f<30)
    
    return np.trapz(PSD[beta_mask], x=f[beta_mask])
    
def optimize_beta_power(trial):
    sn = build_network(trial.suggest_float("poisson_rescale", 0.1 , 2 ),
                       trial.suggest_float("weight_rescale", 0.1, 2))
    trial_beta_power = beta_power(sn)
    del sn
    return trial_beta_power
   
@profile 
def run_optimize():
# %%
	set_verbosity(0) # Turns off C++ outputs

	study = optuna.create_study(direction = 'maximize')
	study.optimize(optimize_beta_power, n_trials = 100, n_jobs = -1, catch=(ValueError, TypeError))
	return study.best_params
# %% [markdown]
# ## Plot best params

# %%
set_verbosity(1)
best_params = {'poisson_rescale': 1.9503052013158302, 'weight_rescale': 1.915076521371698}#run_optimize()
print(best_params)
sn = build_network(1, 1)
print(beta_power(sn))

# %%
from scipy.integrate import simpson

timesteps_per_bin = 10
sampling_frequency = 1e4/timesteps_per_bin
fig, ax = plt.subplots()

binned_spikes = dict()
for pop in sn.populations:
    spikes = sn.populations[pop].get_data()['spikes']
    binned_spikes[pop] = bin_spikes(spikes)
    binned_spikes[pop] = binned_spikes[pop][200:]
    print(f"Mean fire rate {pop} is {np.mean(binned_spikes[pop])/sn.populations[pop].n_neurons :.1f} Hz")
    f, PSD = welch(binned_spikes[pop], 
                   1000, 
                   nperseg=600, 
                   noverlap=150,
                   nfft=None, 
                   scaling='density', 
                   window='hamming')

    norm = simpson(PSD, x=f)
    plt.plot(f, PSD/norm, label=pop)
print(f"F resolution { f[1] - f[0] :.2f} Hz")
# plt.yscale('log')
plt.legend()
plt.ylabel("normalized PSD")
plt.xlabel("Frequency [Hz]")
plt.xlim(0,150)

# %%
for pop in sn.populations:
    plt.plot(binned_spikes[pop]/sn.populations[pop].n_neurons, label=pop)
plt.axhline()
plt.xlabel("Time [ms]")
plt.ylabel("Spiking events per ms per neuron")

# %% [markdown]
# ## Time signal analysis

# %%
from scipy.signal import butter, sosfiltfilt, freqz


def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data
    
fig, ax = plt.subplots()

pop = "GPeTI"

tt = np.linspace(0, 5, len(binned_spikes[pop]))

plt.plot(tt,binned_spikes[pop])
plt.plot(tt, bandpass(binned_spikes[pop], [12, 24], 1000), label="beta")
plt.plot(tt, bandpass(binned_spikes[pop], [30, 140], 1000), label="gamma")
plt.legend()
plt.show()
# %%



