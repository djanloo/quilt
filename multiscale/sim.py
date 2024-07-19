import numpy as np

from quilt.builder import SpikingNetwork, OscillatorNetwork, MultiscaleNetwork
from quilt.interface.base import Projection, set_verbosity

from quilt.utils import firing_rate, bandpass

SPIKING_NET="../bg_analysis/ortone_network.yaml"
NEURONS    = "../bg_analysis/ortone_neurons.yaml"
TRANSDUCERS = "./transducers.yaml"
OSCILLATOR_NET = "../brain_data/connectivity_76.zip"
set_verbosity(2)

# Builds the spiking network
spikenet = SpikingNetwork.from_yaml(SPIKING_NET, NEURONS)
spikenet.monitorize_spikes()
# spikenet.monitorize_states("Albert")
spikenet.build()

# Builds the oscillator network
oscnet = OscillatorNetwork.homogeneous_from_TVB(OSCILLATOR_NET, 
                                            {'oscillator_type':'jansen-rit'}, 
                                            global_weight=0.1, 
                                            conduction_speed=1)
oscnet.build()

multinet = MultiscaleNetwork(spikenet, oscnet)

multinet.add_transducers(TRANSDUCERS)
np.random.seed(1997)

T2O_delays = np.random.uniform(10, 11, size = (multinet.n_transducers, multinet.n_oscillators)).astype(np.float32)
T2O_weights = np.random.uniform(0, 0.1, size = (multinet.n_transducers, multinet.n_oscillators)).astype(np.float32)

# Deletes some links
T2O_weights[(T2O_weights < 0.5)] = 0.0

O2T_delays = np.random.uniform(10, 11, size=(multinet.n_oscillators, multinet.n_transducers)).astype(np.float32)
O2T_weights = np.random.uniform(0, 0.1, size=(multinet.n_oscillators, multinet.n_transducers)).astype(np.float32)
# Deletes some links
O2T_weights[(O2T_weights < 0.5)] = 0.0

T2Oproj = Projection(T2O_weights, T2O_delays)
O2Tproj = Projection(O2T_weights, O2T_delays)

multinet.build_multiscale_projections(T2O=T2Oproj, O2T=O2Tproj)

states = np.random.uniform(0,0.1 , size=6*oscnet.n_oscillators).reshape(oscnet.n_oscillators, 6)

multinet.set_evolution_contextes(dt_short=0.1, dt_long=1.0)
multinet.initialize(states)

print("Printing max times:")
for _ in [O2T_delays, T2O_delays, oscnet.features['connectivity']['delays']]:
    print(np.min(_[~np.isnan(_)]))

T_sec = 2
multinet.run(time=T_sec*1000)

pops = multinet.spiking_network.populations
oscs = multinet.oscillator_network.oscillators


import matplotlib.pyplot as plt
N_osc_plot = 5
fig, axes = plt.subplots(len(pops.keys())+ N_osc_plot, 1, sharex=True)

for i in range(len(pops.keys())):
    axes[i].set_ylabel(list(pops.keys())[i])
    spikes = firing_rate(spikenet, list(pops.keys())[i])
    time = np.linspace(0, T_sec, len(spikes))
    mask = time > 0.0
    axes[i].plot(time[mask], spikes[mask])
    axes[i].plot(time[mask], bandpass(spikes[mask] , [0.1, 12], 1e3), color="red")
    axes[i].plot(time[mask], bandpass(spikes[mask] , [12, 30], 1e3), color="green")
    axes[i].plot(time[mask], bandpass(spikes[mask] , [30, 50], 1e3), color="blue")


for i in range(len(pops), N_osc_plot + len(pops)):
    axes[i].set_ylabel(list(oscs.keys())[i])
    signal = oscs[list(oscs.keys())[i]].history[:, 0]
    time = np.linspace(0, T_sec, len(signal))
    mask = time > 0.0
    axes[i].plot(time[mask], signal[mask])

plt.autoscale(enable=True, axis='y')
# plt.xlim(500, 1000)
plt.show()