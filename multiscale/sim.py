import numpy as np
import matplotlib.pyplot as plt
from quilt.builder import SpikingNetwork, OscillatorNetwork, MultiscaleNetwork
from quilt.interface.base import Projection, set_verbosity

from quilt.utils import firing_rate, bandpass

SPIKING_NET="../bg_analysis/ortone_network.yaml"
NEURONS    = "../bg_analysis/ortone_neurons.yaml"
TRANSDUCERS = "./transducers.yaml"
OSCILLATOR_NET = "../brain_data/connectivity_76.zip"

POPS_PLOT = ["D1", "D2", "GPeTI"]

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
# np.random.seed(1997)

T2O_delays = np.random.uniform(10, 11, size = (multinet.n_transducers, multinet.n_oscillators)).astype(np.float32)
T2O_weights = np.random.uniform(0, 0.2, size = (multinet.n_transducers, multinet.n_oscillators)).astype(np.float32)

# Deletes some links
T2O_weights[(T2O_weights < 0.1)] = 0.0

O2T_delays = np.random.uniform(10, 11, size=(multinet.n_oscillators, multinet.n_transducers)).astype(np.float32)
O2T_weights = np.random.uniform(0, 50.0 , size=(multinet.n_oscillators, multinet.n_transducers)).astype(np.float32)
# Deletes some links
# O2T_weights *= 0
O2T_weights[(O2T_weights < 25)] = 0.0

T2Oproj = Projection(T2O_weights, T2O_delays)
O2Tproj = Projection(O2T_weights, O2T_delays)


multinet.build_multiscale_projections(T2O=T2Oproj, O2T=O2Tproj)

states = np.random.uniform(0,0.1 , size=6*oscnet.n_oscillators).reshape(oscnet.n_oscillators, 6)

multinet.set_evolution_contextes(dt_short=0.1, dt_long=1.0)
multinet.initialize(states)

T_sec = 5
T_burn_in = 0.1
multinet.run(time=T_sec*1000)

pops = multinet.spiking_network.populations
oscs = multinet.oscillator_network.oscillators


N_osc_plot = 4
fig, axes = plt.subplots(len(POPS_PLOT)+ N_osc_plot, 1, sharex=True)

for i in range(len(POPS_PLOT)):
    axes[i].set_ylabel(POPS_PLOT[i])
    spikes = firing_rate(spikenet,POPS_PLOT[i])
    time = np.linspace(0, T_sec, len(spikes))
    mask = time > T_burn_in
    avg_rate = np.mean(spikes[mask])
    axes[i].plot(time[mask], spikes[mask], color="grey", alpha=0.4)
    axes[i].plot(time[mask], avg_rate+bandpass(spikes[mask] , [8, 12], 1e3), color="red")
    axes[i].plot(time[mask], avg_rate+bandpass(spikes[mask] , [12, 30], 1e3), color="green")
    axes[i].plot(time[mask], avg_rate+bandpass(spikes[mask] , [30, 50], 1e3), color="blue")
    axes[i].plot(time[mask], avg_rate+bandpass(spikes[mask] , [50, 150], 1e3), color="purple")

for i in range(len(POPS_PLOT), N_osc_plot + len(POPS_PLOT)):
    axes[i].set_ylabel(list(oscs.keys())[i])
    signal = oscs[list(oscs.keys())[i]].history[:, 0]
    time = np.linspace(0, T_sec, len(signal))
    mask = time > T_burn_in
    axes[i].plot(time[mask], signal[mask])

plt.autoscale(enable=True, axis='y')
# plt.xlim(500, 1000)
# plt.show()

plt.figure(2)

transd_hist = multinet.transducer_histories
fig, axes = plt.subplots(len(transd_hist), 1, sharex=True)
for _, ax in zip(transd_hist, axes):
    ax.plot(_)
plt.show()