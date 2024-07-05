# %% [markdown]
# # Dopamine modulation on BG
# 
# **Observation**: increase in beta-range spectral power is observed without dopamine modulation of CTX input to MSN-D1 and MSN-D2.
# 
# **Method**: I used the network model from Ortone and the model of dopamine modulation from Lindahl(2016). In particular, if an attribute $x$ of the network is subject to dopaminergic modulation, the effect of dopamine levels are modeled by
# 
# $$ x(\alpha) = x_0(1+\chi_x (\alpha -\alpha_0))$$
# where $\alpha$ represents the dopamine level and $x_0$ is the value of the parameter at standard dopamine ($\alpha_0 = 0.8$, Lindahl). In the following, $\chi_x$ will be called generalized susceptibility.
# 
# The **main differences** between Ortone and Lindahl networks are:
# 
# - size of populations
# - synaptic model
# - cortical inputs: no AMPA/NMDA differentiation, different rates and weights
# - STN subthreshold/suprathreshold adaptation parameter ada_a (Ortone: 0, Lindahl 0.3)
# - Rest potential of FSN (Ortone: -80, Lindahl: -64.4). This attribute is negatively susceptible to dopamine, so FSN in Lindahl's network is already dopamine-depleted w.r.t. Ortone's network;
# - SNr synaptic fan-in (from MSN-D1 Ortone: 59, Lindahl: 500; from GPeTI Ortone:25, Lindahl: 32 ). This is not particularly relevant for now since SNr is an output;

# %%
import yaml
from rich import print

NEURONS_FILE = "ortone_neurons.yaml"
NETWORK_FILE = "ortone_network.yaml"
DOPAMINE_FILE = "lindahl_dopamine_susceptibility_noctx_disentangled.yaml"

print("List of dopamine-dependent attributes:")
with open(DOPAMINE_FILE, "r") as dopfile:
    susceptiblities = yaml.safe_load(dopfile)['parametric']['dopamine']

for item in susceptiblities:
    if 'population' in item.keys():
        print(f"Attribute {item['attribute']:15} of population {item['population']:15}: chi = {item['susceptibility']}")
    elif 'projection' in item.keys():
        print(f"Attribute {item['attribute']:10} of projection {item['projection']:15}: chi = {item['susceptibility']}")

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## Parameters of the simulation

# %%
Tlong = 8000  # ms
dt = 0.1      # ms

points_per_bin = 1/dt     # bins 1 ms wide
sampling_frequency = 1e3  # 1 kHz sampling frequency
burn_in_millis = 600      # the first part of the record to discard (ms)

# Params for trials
n_trials = 10
Tshort = 1500 # ms

# Rescaling of populations: Lindahl network is unbearable on my machine
# so populations are rescaled to match sizes from Ortone
populations_scaling = dict()
if NETWORK_FILE == "lindahl_network.yaml":
    populations_scaling = dict( MSN_scale= -0.84,
                                FSN_scale= -0.74,
                                GPe_scale= -0.22,
                                STN_scale= 0.05
                                )

reasonable_firing = dict(FSN=15, D1=1.5, D2=1.5, GPeTI=50, GPeTA=10, STN=16)
# FSN [10-20] Hz 
# D1, D2 [0.5–2.5] Hz
# GPe-TI [40–60] Hz
# GPe-TA [5–15] Hz 
# and STN [12–20] Hz


# %% [markdown]
# ## Utils & plots

# %%
from scipy.signal import butter, sosfiltfilt, freqz, welch
from scipy.integrate import simpson
from scipy import stats

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

# %%
def plot_spectrum(sn, scale="log", pops=None):
    if pops is None:
        pops = sn.populations
    fig, axes = plt.subplots(len(pops), 1, sharex=True)
    for ax, pop in zip(axes, pops):
        f, PSD = get_PSD(sn, pop)
        norm = simpson(PSD, x=f)
        ax.plot(f, PSD/norm, label=pop)
        
        ax.set_ylabel(pop)
        ax.set_yscale(scale)
    # print(f"F resolution { f[1] - f[0] :.2f} Hz")
    # plt.yscale('log')
    # plt.legend()
    # plt.ylabel("normalized PSD")
    plt.xlabel("Frequency [Hz]")
    plt.xlim(10, 30)
    fig.set_figheight(8.5)
    return fig

def plot_signals(sn, pops=None):
    if pops is None:
        pops = sn.populations.keys()
    colors = sns.color_palette("rainbow", 10)

    fig, ax = plt.subplot_mosaic([[pop] for pop in pops],sharex=True, figsize=(8,2*len(pops)))

    for pop in pops:    
        binned_spikes = bin_spikes(sn.populations[pop].get_data('spikes'))
        instantaneous_fr = binned_spikes/sn.populations[pop].n_neurons*1000
        
        tt = np.linspace(0, len(instantaneous_fr)/1000, len(instantaneous_fr))
        ax[pop].plot(tt, instantaneous_fr, color="#c3c3c3", label="raw")
            
        ax[pop].plot(tt,np.mean(instantaneous_fr) + bandpass(instantaneous_fr, [12, 30], sampling_frequency), label=r"$\beta$", color=colors[9])
        ax[pop].plot(tt, np.mean(instantaneous_fr)  + bandpass(instantaneous_fr, [30, 90], sampling_frequency), label=r"$\gamma$", color=colors[2])
        
        ax[pop].set_xlim(6, 7)
        ax[pop].set_ylabel(f"{pop} [Hz]")
        ax[pop].legend()
    
    leg = ax[pops[0]].get_legend()
    for a in ax.values():
        a.legend().remove()
    
    fig.legend(handles=leg.legend_handles, loc='upper right')
    
    ax[pops[-1]].set_xlabel("time [s]")
    return fig


# %% [markdown]
# ## Building the network

# %%
from quilt.interface.spiking import set_verbosity
set_verbosity(1)

# %%
from quilt.builder import NeuronCatalogue, ParametricSpikingNetwork

sn = ParametricSpikingNetwork.from_yaml(NETWORK_FILE, 
                                        NEURONS_FILE,
                                        [DOPAMINE_FILE], 
                                         # DISPERSION_FILE, "dbs.yaml"]
                                       )
sn.monitorize_spikes()
# sn.monitorize_states("GPeTA")

# %% [markdown]
# ## Control case (healthy subject)

# %%
sn.set_parameters(dopamine=1.0)
sn.run(dt=dt, time=10)


exit()
# %%
plot_signals(sn, pops=['D2', 'STN']);
plt.gcf().suptitle('Firing rates @ dopamine = 1');
plt.gca().set_xlim(7, 8)

# %%
plot_spectrum(sn)
plt.gca().set_xlim(0, 500)

# %% [markdown]
# ## Lesioned case (PD)

# %%
sn.set_parameters(dopamine=0.4)
sn.run(dt=dt, time=Tlong)

# %%
plot_signals(sn, pops=['D2', 'STN']);
plt.gcf().suptitle('Firing rates @ dopamine = 0.4');
plt.gca().set_xlim(7, 8)

# %% [markdown]
# ## Comparison

# %%
dopamines = [0.6, 0.7, 0.8, 0.9, 1]

nets = []
for dopamine in dopamines:
    nets.append(ParametricSpikingNetwork.from_yaml(NETWORK_FILE, 
                                            NEURONS_FILE,
                                            [DOPAMINE_FILE]))
    nets[-1].set_parameters(dopamine=dopamine)
    nets[-1].monitorize_spikes()

# %%
for net in nets:
    net.run(dt=dt, time=Tlong)

# %%
fig, ax = plt.subplots()
D2_colors = sns.cubehelix_palette(start=0, rot=0, dark=0.3, light=0.8, reverse=False, n_colors=len(nets), hue=1)
STN_colors = sns.cubehelix_palette(start=1.3, rot=0, dark=0.3, light=0.8, reverse=False, n_colors=len(nets), hue=1)

sns.set_style("ticks")

for i in range(len(nets)):
    ax.plot(*get_PSD(nets[i], "D2", frequency_resolution=0.5, smooth=True), label=dopamines[i], color=D2_colors[i])

ax.set_yscale('log')
ax.set_xlim(10, 30)
plt.legend(title="dopamine")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("D2 - PSD")
ax.set_ylim(1e-3, 2e2)
plt.savefig("images/D2-PSD.pdf")

# %%
fig, ax = plt.subplots()
STN_colors = sns.cubehelix_palette(start=1.3, rot=0, dark=0.3, light=0.8, reverse=False, n_colors=len(nets), hue=1)

sns.set_style("ticks")

for i in range(len(nets)):
    ax.plot(*get_PSD(nets[i], "STN", frequency_resolution=0.5, smooth=True), label=dopamines[i], color=STN_colors[i])
    # ax.plot(*get_PSD(nets[i], "STN"), label="STN", color=STN_colors[i])
ax.set_yscale('log')
ax.set_xlim(10, 2000)
plt.legend(title="dopamine")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("STN - PSD")
ax.set_ylim(1e-3, 2e2)
plt.savefig("images/STN-PSD.pdf")

# %% [markdown]
# ## Trend

# %%
from rich.progress import track

set_verbosity(0)
beta_powers = dict()
fmaxs = dict()
beta_powers_normalized = dict()
entropies = dict()

for pop in sn.spike_monitored_pops:
    fmaxs[pop] = []
    beta_powers[pop] = []
    beta_powers_normalized[pop] = []
    entropies[pop] = []
    
dopamine_levels = np.linspace(0.2, 0.9, 20)

for dopamine in track(dopamine_levels, total=len(dopamine_levels)):
    sn.set_parameters(dopamine=dopamine)
    sn.run(dt=dt, time=1600)
    for pop in sn.populations.keys():
        fmax, normbeta, beta, entropy = beta_power(sn, pop)
        fmaxs[pop] += [fmax]
        beta_powers_normalized[pop] += [normbeta]
        beta_powers[pop] += [beta]
        entropies[pop] += [entropy]

# %%
for pop in ["D2", "STN"]:
    plt.plot(dopamine_levels, beta_powers[pop], label=pop)
plt.ylabel("Beta power")
plt.xlabel("Dopamine")
plt.yscale('log')
plt.legend()

# %%
for pop in fmaxs:
    plt.plot(dopamine_levels, fmaxs[pop], label=pop)
# plt.yscale('log')
plt.ylabel("Spectral mode [Hz]")
plt.xlabel("Dopamine")
plt.legend()
plt.ylim(13, 17)

# %%
for pop in ["D2", "STN"]:
    plt.plot(dopamine_levels, entropies[pop], label=pop)
# plt.yscale('log')
plt.xlabel("Dopamine")
plt.ylabel("Spectral Entropy")
plt.legend()

# %%
for pop in ["D2", "STN"]:
    plt.plot(dopamine_levels, beta_powers_normalized[pop], label=pop)
# plt.yscale('log')
plt.ylabel("Beta power [a.u.]")
plt.xlabel("Dopamine")
plt.legend()

# %%
for pop in beta_powers_normalized:
    plt.plot(entropies[pop], beta_powers_normalized[pop], label=pop, ls="", marker=".")
plt.xlabel("Spectral entropy")
plt.ylabel("Beta power [a.u.]")
plt.legend()

# %% [markdown]
# ## plot 2D

# %%
from scipy.integrate import simpson
def bin_spectrum(f, PSD, n_bins=20, fmax=90):
    norm = simpson(PSD[f<fmax], x=f[f<fmax])
    
    binned_spectrum = np.zeros(n_bins)
    frequency_bin_width = fmax/n_bins
    for i in range(n_bins):
        f_mask = (f > i*frequency_bin_width)&(f < (i+1)*frequency_bin_width)
        # binned_spectrum[i] = simpson(PSD[f_mask], x=f[f_mask])/norm
    return binned_spectrum

# %%
from rich.progress import track

set_verbosity(0)

spectral_resolution = 2.5
N_dopamine_levels = int(90/spectral_resolution)
N_spectral_bins = int(90/spectral_resolution)
dopamine_levels = np.linspace(0.2, 1.0, N_dopamine_levels)

spectra = dict()
for pop in sn.features_dict['populations']:
    spectra[pop] = np.zeros((N_dopamine_levels, N_spectral_bins))

for i in track(range(N_dopamine_levels), total=len(dopamine_levels)):
    sn.set_parameters(dopamine=dopamine_levels[i], 
                      delay_dispersion=-1, 
                      **populations_scaling)
    sn.run(dt=dt, time=3 * sampling_frequency/spectral_resolution + burn_in_millis ) # Uses 3 complete windows at minimal resolution
    for pop in sn.populations.keys():
        f, PSD = get_PSD(sn, pop, frequency_resolution=spectral_resolution)
        plt.show()
        spectra[pop][i] = PSD[f<90]
    

# %%
X, Y = np.meshgrid(dopamine_levels, np.linspace(0, 90, int(90/spectral_resolution)))
# pop = "GPeTA"
for pop in sn.populations:
    plt.pcolormesh(X, Y, np.log10(spectra[pop].T + 1e-10))
    plt.xlabel("Dopamine")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label = "Log PSD")
    plt.title(f"{pop} - PSD")
    plt.savefig(f"images/logPSD_{pop}.pdf")
    plt.show()

# %% [markdown]
# ## Statistical features of the signal

# %%
def plot_spikecount_distrib(sn):
    for pop in sn.populations:
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.set_title(pop)
        spikes = sn.populations[pop].get_data('spikes')
        spikes = spikes[5000:]
        ax2.plot(spikes)
        # ax2.plot(bin_spikes(spikes))
        from scipy.special import factorial
        nn = np.arange(np.min(spikes), np.max(spikes))
        histogram, bins = np.histogram(spikes, bins=nn, density=True)
        ax1.step(bins[:-1], histogram, where='mid', marker=".", label="Spike count")
        l = np.mean(spikes)
        print(l)
        ax1.step(nn, l**nn*np.exp(-l)/factorial(nn.astype(int)), marker=".", where='mid', label=r"Poisson ($\lambda$ = $\langle N \rangle$)")
        ax1.legend()
        plt.show()

# %%
sn.set_parameters(dopamine=1.0)
sn.run(dt=dt, time=Tshort)

# %%
plot_spikecount_distrib(sn)

# %%
sn.set_parameters(dopamine=0.4)
sn.run(dt=dt, time=Tshort)

# %%
plot_spikecount_distrib(sn)

# %% [markdown]
# ## Estimating rate functions

# %%
spikes = sn.populations['GPeTI'].get_data('spikes')
tt = np.linspace(0, Tshort, len(spikes))

mask = tt >1000

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(tt[mask].reshape(-1,1), np.cumsum(spikes)[mask])

m_fit = model.coef_[0]
b_fit = model.intercept_ 
L = np.cumsum(spikes)[mask] - m_fit*tt[mask]
plt.plot(tt[mask], 0.01*L)
for j in range(15):
    for i in range(1, len(tt[mask])-1):
        L[i] = 0.5*(L[i+1] + L[i-1])
plt.plot(tt[mask], 0.01*L)
plt.plot(tt[mask][:-1], 0.1*np.diff(L)/np.diff(tt[mask]), alpha=0.8)
# plt.ylim(0, 0.25)

# %%
k = 5 
plt.scatter(spikes[:-k]+np.random.normal(0, 0.1, size=len(spikes[:-k])), spikes[k:]+np.random.normal(0, 0.1, size=len(spikes[:-k])), s=1)

# %%
from scipy.signal import correlate
gg = correlate(spikes, spikes, mode='full')/np.std(spikes)**2
gg = gg[len(gg)//2:]
gg /= gg[0]
plt.plot(gg)

# %%
cvs = []
times = []
k = 100
for bunch, ttt in zip(np.split(spikes[:-(len(spikes)%k)], k),
                     np.split(tt[:-(len(spikes)%k)], k)):
    times.append(0.5*(ttt[0] + ttt[-1]))
    cvs.append(np.std(bunch)**2/np.mean(bunch)**2)
plt.step(times, cvs, where='mid')
plt.plot(tt, spikes/10, alpha=0.3)
plt.xlim(0, 1000)

# %% [markdown]
# ## Relevance analysis

# %%
import yaml

with open(DOPAMINE_FILE, "r") as f:
    parameters_dict = yaml.safe_load(f)

new_parameters_dict = dict(parameters=dict(), parametric=dict())
counter = 0
for parametric in parameters_dict['parametric']['dopamine']:
    if 'population' in parametric:
        name = parametric['population'].replace(" ", "") + "_" + parametric['attribute'] 
    elif 'projection' in parametric:
        name = parametric['projection'].replace(" ", "") + "_" + parametric['attribute'] 
        
    new_parameters_dict["parameters"].update({name : {'min': 0, 'max': 1, 'shift': 0.8}})
    new_parameters_dict["parametric"][name] =[parametric]

# %%
from quilt.builder import ParametricSpikingNetwork
net = ParametricSpikingNetwork.from_dict(new_parameters_dict, network_file=NETWORK_FILE, neuron_file=NEURONS_FILE)
net.monitorize_spikes()

# %% [markdown]
# #### General sampling

# %%
import pandas as pd
from rich.progress import track
N = 200
eps = 1.0
alpha0 = 0.5
burn_in_millis=1000

set_verbosity(0)

results = pd.DataFrame()
for i in track(range(N)):
    dopamine_array = dict()
    
    for parameter in new_parameters_dict['parameters']:
        dopamine_array[parameter] = alpha0 + eps/2*np.random.uniform(-1, 1)
    net.set_parameters(**dopamine_array)

    net.run(dt=0.1, time=5000)

    row = dopamine_array.copy()

    for pop in net.populations.keys():
        fmax, fmean, norm_beta_pow, beta_pow, entropy = beta_power(net, pop)
        row[f"{pop}_norm_beta_power"] = norm_beta_pow
        row[f"{pop}_beta_power"] = beta_pow
        row[f"{pop}_fmax"] = fmax
        row[f"{pop}_entropy"] = entropy
        row[f"{pop}_fmean"] = fmean
    results = pd.concat([results, pd.DataFrame(row, index=[i])])

results.to_csv("multidim_dopamine_6.csv")
display(results)

# %% [markdown]
# #### Around diagonal sampling

# %%
import pandas as pd
from rich.progress import track
N = 4
eps = 0.05
alpha0s = np.linspace(0, 1, 30)
burn_in_millis=1000

set_verbosity(0)

results = pd.DataFrame()


for a0 in track(alpha0s):
    for i in range(N):
        dopamine_array = dict()

        for parameter in new_parameters_dict['parameters']:
            proposed_dopamine = a0 + eps/2*np.random.uniform(-1, 1)
            while proposed_dopamine < 0 or proposed_dopamine > 1:
                proposed_dopamine = a0 + eps/2*np.random.uniform(-1, 1)
            dopamine_array[parameter] = proposed_dopamine
        net.set_parameters(**dopamine_array)

        net.run(dt=0.1, time=5000)

        row = dopamine_array.copy()

        for pop in net.populations.keys():
            pop_result = beta_power(net, pop)
            for key in pop_result:
                row[f"{pop}_{key}"] = pop_result[key]
            row[f"{pop}_mean_fr"] = firing_rate(net, pop)
        results = pd.concat([results, pd.DataFrame(row, index=[len(results)])])

results.to_csv("multidim_dopamine_around_diagonal_5.csv")
display(results)

# %% [markdown]
# ### On-diagonal sampling

# %%
import pandas as pd
from rich.progress import track

alpha0s = np.linspace(0, 1, 30)
N = 4
burn_in_millis=1000

set_verbosity(0)
results = pd.DataFrame()

for a0 in track(alpha0s):
    for i in range(N):
        dopamine_array = dict()
    
        for parameter in new_parameters_dict['parameters']:
            dopamine_array[parameter] = a0
        net.set_parameters(**dopamine_array)
    
        net.run(dt=0.1, time=5000)
    
        row = dopamine_array.copy()
    
        for pop in net.populations.keys():
            pop_result = beta_power(net, pop)
            for key in pop_result:
                row[f"{pop}_{key}"] = pop_result[key]
            row[f"{pop}_mean_fr"] = firing_rate(net, pop)
        results = pd.concat([results, pd.DataFrame(row, index=[len(results)])])

results.to_csv("multidim_dopamine_diagonal_4_trials.csv")
display(results)

# %% [markdown]
# ### Data

# %%
import numpy as np
from matplotlib import pyplot as plt 
import optuna 
import seaborn as sns
import pandas as pd

# %%
populations = ["D1", "D2", "FSN", "STN", "GPeTI", "GPeTA", "SNR"]

# data_1 = pd.read_csv("multidim_dopamine_around_diagonal_3.csv")

# for pop in populations:
#     data_1[f"{pop}_fmean"] /= 2

data_2 = pd.read_csv("multidim_dopamine_around_diagonal_4.csv")
data_3 = pd.read_csv("multidim_dopamine_diagonal_4_trials.csv")

data = pd.concat([data_2, data_3])
# data = data_3.copy()

try:
    data = data.drop(columns = ["Unnamed: 0"])
    # data = data.drop(columns = ["level_0"])
except KeyError:
    pass

data = data.reset_index(drop=True)

features = []
observables = []

for col in data.columns:
    if "power" in col or "entropy" in col or "fmax" in col or "fmean" in col or "mean_fr" in col:
        observables.append(col)
    else:
        features.append(col)

# data = data.loc[np.mean(data[features], axis=1) <0.8]


X = data[features].to_numpy()
analysis_feature = "STN_beta_power"
y = data[analysis_feature].to_numpy()

data['alpha'] = np.mean(X, axis=1)

# %%
from umap import UMAP

red = UMAP(n_neighbors=100)
embedding = red.fit_transform(X)

# %%
plt.scatter(*embedding.T, c=data.STN_beta_power, s=20)
plt.colorbar()

# %%
fmean_data = data[[o for o in observables if "fmean" in o]]
order = [f"{pop}_fmean" for pop in ["D1", "D2", "GPeTI", "STN", "GPeTA", "FSN"]]
fmean_data=fmean_data.reindex(columns=order)
sns.pairplot(fmean_data)

# %%
mean_fr_data = data[[o for o in observables if "mean_fr" in o]]
order = [f"{pop}_mean_fr" for pop in ["D1", "D2", "GPeTI", "STN", "GPeTA", "FSN"]]
mean_fr_data=mean_fr_data.reindex(columns=order)
sns.pairplot(mean_fr_data)

# %%
plt.close()
# plt.scatter(data.alpha, data.STN_beta_power, c=data.STN_fmean, s=10)
# plt.scatter(data.alpha, data.STN_low_gamma_power, c=data.STN_fmean, s=10)

plt.scatter(data.alpha, data.STN_low_gamma_power/data.STN_beta_power, c=data.GPeTI_fmean, s=10)

plt.colorbar()

# %%
pop = "FSN"
plt.scatter(data.alpha, data[f"{pop}_beta_power"],s=10)
plt.scatter(data.alpha, data[f"{pop}_low_gamma_power"], s=10)
# plt.scatter(data.alpha, data[f"{pop}_high_gamma_power"], s=10)

# %%
pop = "STN"
plt.scatter(data.alpha, data[f"{pop}_beta_power"],s=10)
plt.scatter(data.alpha, data[f"{pop}_low_gamma_power"], s=10)
plt.scatter(data.alpha, data[f"{pop}_high_gamma_power"], s=10)

# %%
from matplotlib.colors import ListedColormap

pop1 = "STN"
pop2 = "GPeTI"

cmap = ListedColormap(sns.color_palette("rainbow_r", 4).as_hex())
for i in range(len(populations)):
    for j in range(i):
        pop1 = populations[i]
        pop2 = populations[j]
        comparison = data[features + [f"{pop1}_fmean", f"{pop2}_fmean"]]
        comparison.loc[:, "difference"] =  comparison.loc[:, f"{pop1}_fmean"].to_numpy() - comparison[f"{pop2}_fmean"].to_numpy()
        comparison["alpha"] = np.mean(comparison[features], axis=1)
        # plt.scatter(comparison.alpha, np.abs(comparison.difference))
        
        size = data[f"{pop1}_beta_power"] + data[f"{pop2}_beta_power"]
        size = 50*MinMaxScaler().fit_transform(size.to_numpy().reshape(-1,1)) + 5
        
        fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
        
        plt.scatter(comparison[f"{pop1}_fmean"], comparison[f"{pop2}_fmean"], 
                    c= comparison.alpha, 
                    s=size, 
                    cmap=cmap
        )
        cbar = plt.colorbar(shrink=0.73)
        cbar.set_label("Average dopamine")
        plt.xlabel(f"{pop1} mean frequency")
        plt.ylabel(f"{pop2} mean frequency")
        
        allvals = np.concatenate((comparison[f"{pop1}_fmean"],comparison[f"{pop2}_fmean"]))
        
        plt.plot([np.min(allvals), np.max(allvals)], 
                 [np.min(allvals), np.max(allvals)],
                color="k", ls=":")
        ax.set_aspect("equal")
        
        # Creiamo una legenda personalizzata
        plt.scatter([], [], color='gray', s=20, label='High beta power')  # Pallino grigio grande
        plt.scatter([], [], color='gray', s=5, label='Low beta power')    # Pallino grigio piccolo

        # plt.scatter(data[f"{pop1}_fmax"], data[f"{pop2}_fmax"], s=2, color="k")
        # Aggiungiamo la legenda al plot
        plt.legend()
        plt.savefig(f"images/loops_syncronization_{pop1}{pop2}.pdf", bbox_inches="tight")

# %%
from matplotlib.colors import ListedColormap

pop1 = "STN"
pop2 = "GPeTI"

cmap = ListedColormap(sns.color_palette("rainbow_r", 4).as_hex())
for i in range(len(populations)):
    for j in range(i):
        pop1 = populations[i]
        pop2 = populations[j]
        comparison = data[features + [f"{pop1}_mean_fr", f"{pop2}_mean_fr"]]
        comparison.loc[:, "difference"] =  comparison.loc[:, f"{pop1}_mean_fr"].to_numpy() - comparison[f"{pop2}_mean_fr"].to_numpy()
        comparison["alpha"] = np.mean(comparison[features], axis=1)
        # plt.scatter(comparison.alpha, np.abs(comparison.difference))
        
        size = data[f"{pop1}_beta_power"] + data[f"{pop2}_beta_power"]
        size = 50*MinMaxScaler().fit_transform(size.to_numpy().reshape(-1,1)) + 5
        
        fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
        
        plt.scatter(comparison[f"{pop1}_mean_fr"], comparison[f"{pop2}_mean_fr"], 
                    c= comparison.alpha, 
                    s=size, 
                    cmap=cmap
        )
        cbar = plt.colorbar(shrink=0.73)

        try:
            plt.scatter([reasonable_firing[pop1]], [reasonable_firing[pop2]], s=100, marker="X", color="k")
        except KeyError:
            pass
        cbar.set_label("Average dopamine")
        plt.xlabel(f"{pop1} mean firing rate")
        plt.ylabel(f"{pop2} mean firing rate")
        
        # allvals = np.concatenate((comparison[f"{pop1}_mean_fr"],comparison[f"{pop2}_mean_fr"]))
        
        # plt.plot([np.min(allvals), np.max(allvals)], 
        #          [np.min(allvals), np.max(allvals)],
        #         color="k", ls=":")
        # ax.set_aspect("equal")
        
        # Creiamo una legenda personalizzata
        plt.scatter([], [], color='gray', s=20, label='High beta power')  # Pallino grigio grande
        plt.scatter([], [], color='gray', s=5, label='Low beta power')    # Pallino grigio piccolo

        # plt.scatter(data[f"{pop1}_fmax"], data[f"{pop2}_fmax"], s=2, color="k")
        # Aggiungiamo la legenda al plot
        plt.legend()
        plt.savefig(f"images/firing_rates_{pop1}{pop2}.pdf", bbox_inches="tight")

# %%
fmean_data = data[[o for o in observables if "fmean" in o]]
order = [f"{pop}_fmean" for pop in ["D1", "D2", "GPeTI", "STN", "GPeTA", "FSN"]]
fmean_data=fmean_data.reindex(columns=order)

fmean_difference = pd.DataFrame(columns=["D1", "D2", "GPeTI", "FSN", "STN", "GPeTA"])
for pop1 in fmean_difference.columns:
    row = dict()
    for pop2 in fmean_difference.columns:
        row[pop2] = np.mean(np.abs(fmean_data[f"{pop1}_fmean"] - fmean_data[f"{pop2}_fmean"]))
    row = pd.DataFrame(row, index=[pop1])
    fmean_difference = pd.concat([fmean_difference, row])
fmean_difference

# fmean_difference.index=["D1", "D2", "GPeTI", "STN", "GPeTA", "FSN"]
# fmean_data
# plt.gca().set_aspect("equal")
sns.heatmap(fmean_difference)

# %%
from scipy.stats import  spearmanr

populations = ["D1", "D2", "FSN", "STN", "GPeTI", "GPeTA", "SNR"]
beta_power_observables = [f"{pop}_fmean" for pop in populations]

M = len(beta_power_observables)
spearman_beta = np.zeros((M, M))

for i in range(M):
    for j in range(M):
        spearman_beta[i,j] = spearmanr(data[beta_power_observables[i]], data[beta_power_observables[j]])[0]

plt.matshow(spearman_beta)
plt.yticks(range(M), labels=beta_power_observables);
plt.xticks(range(M), labels=beta_power_observables, rotation=90);
plt.colorbar()
# sns.heatmap|(spearman_beta, annot=True)


# %% [markdown]
# ### Neural networks

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# %%
from sklearn.model_selection import train_test_split


train_ids, val_ids = train_test_split(np.arange(len(X)),  test_size=0.1)

pop = "STN"
analysis_feature1 = "beta_power"
analysis_feature2 = "low_gamma_power"
analysis_feature3 = "high_gamma_power"

y = data[f"{pop}_{analysis_feature2}"].to_numpy()

# %%
from sklearn.metrics import mean_squared_error
import optuna

def create_model(params):

    model = Sequential()
    model.add(Dense(X.shape[1], activation='linear'))

    for i in range(params["n_hidden"]):
        model.add(Dense(params["n_units"], activation='softplus'))

    model.add(Dense(1, activation="linear"))
    model.compile(loss='mean_squared_error',
                  optimizer="adam")
    return model


# objective function to be minimized
def objective_fun(trial):
    
    params = dict(n_units=trial.suggest_int("n_units", 64, 256),
                 n_hidden=trial.suggest_int("n_hidden", 8, 16),
#                  learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-1)
                 )
    
    model = create_model(params)
    model.fit(X[train_ids], y[train_ids], 
              epochs=trial.suggest_int("epochs", 150, 200), 
              batch_size=10, 
              validation_data=(X[val_ids], y[val_ids]), 
              verbose=False)
    y_pred = model.predict(X[val_ids], verbose=False)

    error = mean_squared_error(y_pred, y[val_ids])
    return error


study = optuna.create_study(direction="minimize")
study.optimize(objective_fun, n_trials=100, n_jobs=-1)

# %%
params = dict(n_hidden=10, n_units=128)

pop = "GPeTI"
params = study.best_params
nn1 = create_model(params)
nn1.fit(X, data[f"{pop}_{analysis_feature1}"], epochs=300, batch_size=100, verbose=False)

nn2 = create_model(params)
nn2.fit(X, data[f"{pop}_{analysis_feature2}"], epochs=300, batch_size=100, verbose=False)

nn3 = create_model(params)
nn3.fit(X, data[f"{pop}_{analysis_feature3}"], epochs=300, batch_size=100, verbose=False)


# %%
plt.scatter(data[f"{pop}_{analysis_feature1}"], nn1.predict(X), s=10)
plt.scatter(data[f"{pop}_{analysis_feature2}"], nn2.predict(X), s=10)
plt.scatter(data[f"{pop}_{analysis_feature3}"], nn3.predict(X), s=10)

# %%
XX = np.repeat(np.linspace(0.0, 1.0, 100), X.shape[1]).reshape(-1,X.shape[1] )

# plt.plot(XX[:,0], nn1.predict(XX), color="k")
# plt.plot(XX[:,0], nn2.predict(XX), color="k")
# plt.plot(XX[:,0], nn3.predict(XX), color="k")
plt.close()

importance = np.exp(-np.std(X, axis=1))
from sklearn.preprocessing import MinMaxScaler
importance = MinMaxScaler().fit_transform(importance.reshape(-1,1))

plt.scatter(np.mean(X, axis=1), data[f"{pop}_{analysis_feature2}"]/data[f"{pop}_{analysis_feature1}"] , s=10*(importance+0.2), alpha=1, c=importance, cmap="plasma_r")
# plt.scatter(np.mean(X, axis=1), data[f"{pop}_{analysis_feature2}"], s=10*(importance+0.2), alpha=1, c=importance, cmap="rainbow_r")
# plt.scatter(np.mean(X, axis=1), data[f"{pop}_{analysis_feature3}"], s=10*(importance+0.2), alpha=1, c=importance, cmap="plasma_r")

plt.xlabel("Average dopamine")
plt.ylabel(pop)

# %%
analysis_feature1

# %% [markdown]
# #### Local relevance

# %%


# %%
from sklearn.linear_model import LinearRegression

# %%
N = 300
l = 0.002

relevance_df = pd.DataFrame()
for alpha0 in np.linspace(0, 1, 50):
    XX = np.random.normal(alpha0, l, size=(N, X.shape[1]))

    yy = nn.predict(XX)

    local_model = LinearRegression()
    local_model.fit(XX, yy);
    local_model.coef_

    row = dict(alpha=[alpha0]*len(features), 
               feature=features, 
               relevance=local_model.coef_[0],
               function_value=[np.mean(y)]*len(features)
               )
    row = pd.DataFrame(row)
    relevance_df = pd.concat([relevance_df, row])
# relevance_df = relevance_df.sort_values(by="relevance")
relevance_df

# %%
# Remove the less relevant
total_relevances = []
for feat in features:
    dd = relevance_df.loc[ relevance_df.feature == feat ]
    tot_rel = np.sum( dd.relevance.to_numpy()**2)
    total_relevances += [tot_rel]

best_rel_df = pd.DataFrame(dict(feature = features, tot_relevance = total_relevances)).sort_values(by="tot_relevance")
best_rel_df

# %%
# select best
best_feats = best_rel_df.sort_values('tot_relevance',ascending = False).head(6)
best_feats = best_feats.feature.values
best_feats

# %%
plt.style.use("../style.mplstyle")
subset = relevance_df.loc[relevance_df['feature'].isin(best_feats)]
color_dict = {"GPeTA_E_l":"yellow",
              "FSN->D2_connectivity": "purple",
              "FSN->D2_connectivity": "red",
              "D2->GPeTI_weight": "blue",
              "GPeTI->GPeTI_weight": "orange",
              "GPeTI->FSN_weight": "green",
              "D2->D2_connectivity": "cyan"
              }

for bf in best_feats:
    subset = relevance_df[relevance_df.feature == bf]
    plt.plot(subset.alpha, subset.relevance, c=color_dict.get(bf, "k"), label=bf)

summ = np.zeros(len(relevance_df[relevance_df.feature == features[0]]))
summ_bests = np.zeros(len(relevance_df[relevance_df.feature == features[0]]))
summ_pippe = np.zeros(len(relevance_df[relevance_df.feature == features[0]]))

for feat in features:
    subset = relevance_df[relevance_df.feature == feat]
    summ += subset.relevance.to_numpy()
    if feat not in best_feats:
        plt.plot(subset.alpha, subset.relevance, color="k", alpha=0.2)
        summ_pippe += subset.relevance.to_numpy()

    if feat in best_feats:
        summ_bests += subset.relevance.to_numpy()

plt.plot(subset.alpha, summ, color="k", ls="--", label="Total Sum")
# plt.plot(subset.alpha, summ_bests, color="k", ls=":", label="Summ best")
# plt.plot(subset.alpha, summ_pippe, color="r", ls="--", label="Summ best")

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel(r"Dopamine $\alpha$ [a.u.]")
plt.ylabel(r"Sensitivity $b_i$")

plt.savefig("relevance_alpha.pdf", bbox_inches='tight')

# %%
mask = (subset.alpha > 0.3 ) & (subset.alpha < 0.6)
aa = subset.alpha[mask].to_numpy()
bb = summ[mask]

aa[np.argmax(bb)]

# %%
plt.style.use("../style.mplstyle")
subset = relevance_df.loc[relevance_df['feature'].isin(best_feats)]
color_dict = {"GPeTA_E_l":"yellow",
              "FSN->D2_connectivity": "purple",
              "FSN->D2_connectivity": "red",
              "D2->GPeTI_weight": "blue",
              "GPeTI->GPeTI_weight": "orange",
              "GPeTI->FSN_weight": "green",
              "D2->D2_connectivity": "cyan"
              }

for bf in best_feats:
    subset = relevance_df[relevance_df.feature == bf]
    plt.plot(subset.alpha, subset.relevance/new_parameters_dict['parametric'][bf][0]['susceptibility'], 
             c=color_dict.get(bf, "k"), 
             label=bf)

for feat in features:
    if feat not in best_feats:
        subset = relevance_df[relevance_df.feature == feat]
        plt.plot(subset.alpha, subset.relevance/new_parameters_dict['parametric'][feat][0]['susceptibility']/subset.function_value, color="k", alpha=0.1)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel(r"Dopamine $\alpha$ [a.u.]")
plt.ylabel(r"$\epsilon_S /\epsilon_p$")
plt.savefig("relevance_alpha_weighted.pdf", bbox_inches='tight')

# %% [markdown]
# ### Model independent

# %%
def gradient_from_irregular(points, function_evaluations):
    M = len(points)     # Number of points
    L = int(M*(M-1)/2)  # Number of couples
    dim = points.shape[1]
    
    Delta = np.zeros((L, X.shape[1]))
    B = np.zeros(L)
    # print(f"Estimating gradient on {L} couples")
    couples = 0
    couples_dict = dict()
    for i in range(M*M):
        couple_index = i + 1
        point_1 = couple_index // M
        point_2 = couple_index % M
        
        if point_1 >= point_2:
            continue
        
        for dim in range(X.shape[1]):
            Delta[couples, dim] = X[point_2, dim] - X[point_1, dim] 
        norm = np.sqrt(np.sum(Delta[couples]**2))
        
        Delta[couples] /= norm
        B[couples] = (function_evaluations[point_2] - function_evaluations[point_1])/norm
        
        couples_dict[couples] = (point_1, point_2)
        couples +=1

    # Estimate the gradient through inverse
    invmat = np.matmul( np.linalg.inv( np.matmul( Delta.T , Delta)), Delta.T)
    grad = invmat.dot(B)
    return grad

# %%
from sklearn.linear_model import LinearRegression
grad_df = pd.DataFrame()
for alpha_0 in np.linspace(0, 1, 50):
    X = data[features].to_numpy()
    
    distances = np.sqrt( np.sum((X - alpha_0)**2, axis=1))
    is_local = np.argsort(distances)[:60]
    
    data_local = data.loc[is_local]
    X_local = X[is_local]
    y_local = y[is_local]

    # print(np.min(np.mean(X_local, axis=1)),np.max(np.mean(X_local, axis=1)))

    grad = gradient_from_irregular(X_local, y_local)
    # model = LinearRegression()
    # model.fit(X_local, y_local)
    # grad = model.coef_
    
    subdf = pd.DataFrame([grad], columns=features)
    subdf["alpha"] = alpha_0
    grad_df = pd.concat([grad_df, subdf])
grad_df

# %%
for coll in grad_df.columns:
    plt.plot(grad_df.alpha, grad_df[coll], alpha=0.1, color="k")
plt.plot(grad_df.alpha, grad_df.drop(columns=["alpha"]).sum(axis=1).to_numpy(), color="k")

# %%
grad = gradient_from_irregular(X, y)
param_sensitivity = np.zeros(grad.shape)
for i, feat in enumerate(features):
    param_sensitivity[i] = grad[i]/new_parameters_dict['parametric'][feat][0]['susceptibility']

# %%
grad_df = pd.DataFrame(dict(Sensitivity=grad, 
                            Feature=features,
                            ParSens=param_sensitivity,
                            abs_sensitivity=np.abs(grad)))
grad_df = grad_df.nlargest(10, "abs_sensitivity")
grad_df = grad_df.sort_values(by="Sensitivity")

xx = np.max(np.abs(grad_df.Sensitivity))/2
sns.barplot(data=grad_df, y="Feature", x="Sensitivity", hue="Sensitivity", palette="coolwarm_r", legend=False)
plt.annotate(r"Increase $S_\beta$" + "\nwhen dopamine lacks", (-xx, len(grad_df)//2), va="center", ha="center")
plt.annotate(r"Decrease $S_\beta$" + "\nwhen dopamine lacks", (xx, len(grad_df)//2), va="center", ha="center")
plt.savefig("global_relevance.pdf",bbox_inches='tight' )

# %%
grad_df = pd.DataFrame(dict(Sensitivity=grad, 
                            Feature=features,
                            ParSens=param_sensitivity,
                            abs_sensitivity=np.abs(grad),
                            abs_parsens=np.abs(param_sensitivity)))
grad_df = grad_df.nlargest(10, "abs_parsens")
grad_df = grad_df.sort_values(by="ParSens")

xx = np.max(np.abs(grad_df.ParSens))/2

bp = sns.barplot(data=grad_df, y="Feature", x="ParSens", hue="ParSens", palette="coolwarm_r", legend=False)
bp.set_xlabel("Parameter Sensitivity")

# norm = plt.Normalize(grad_df.Sensitivity.min(), grad_df.Sensitivity.max())
# sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
# cbar = ax.figure.colorbar(sm, ax=bp)
# cbar.set_label("Dopamine sensitivity")


# %%



