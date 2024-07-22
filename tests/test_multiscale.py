from quilt.builder import MultiscaleNetwork, SpikingNetwork, OscillatorNetwork
from quilt.interface.base import Projection
from quilt.utils import bin_spikes
import numpy as np

from quilt.interface.base import set_verbosity

set_verbosity(4)

TEST_NET="tests/test_spiking.yaml"
TEST_NEURONS    = "tests/test_neurons.yaml"
TEST_TRANSDUCERS = "tests/test_transducers.yaml"

def test_vanilla_init():
    # Builds the spiking network
    spikenet = SpikingNetwork.from_yaml(TEST_NET, TEST_NEURONS)
    spikenet.build()

    # Builds the oscillator network
    oscnet = OscillatorNetwork.homogeneous_from_TVB('brain_data/connectivity_76.zip', 
                                             {'oscillator_type':'jansen-rit'}, 
                                             global_weight=0.1, 
                                             conduction_speed=0.5)
    oscnet.build()

    multinet = MultiscaleNetwork(spikenet, oscnet)

def test_transducers_from_list():

    # Builds the spiking network
    spikenet = SpikingNetwork.from_yaml(TEST_NET, TEST_NEURONS)
    spikenet.build()

    # Builds the oscillator network
    oscnet = OscillatorNetwork.homogeneous_from_TVB('brain_data/connectivity_76.zip', 
                                             {'oscillator_type':'jansen-rit'}, 
                                             global_weight=0.1, 
                                             conduction_speed=0.5)
    oscnet.build()

    multinet = MultiscaleNetwork(spikenet, oscnet)

    list_of_transducers = [dict(name="dummy", 
                                population="Albert", 
                                generation_window=10, 
                                initialization_rate=100, 
                                weight=0.1,
                                weight_delta=0
                                )
                        ]
    multinet.add_transducers(list_of_transducers)

def test_transducers_from_yaml():

    # Builds the spiking network
    spikenet = SpikingNetwork.from_yaml(TEST_NET, TEST_NEURONS)
    spikenet.build()

    # Builds the oscillator network
    oscnet = OscillatorNetwork.homogeneous_from_TVB('brain_data/connectivity_76.zip', 
                                             {'oscillator_type':'jansen-rit'}, 
                                             global_weight=0.1, 
                                             conduction_speed=0.5)
    oscnet.build()

    multinet = MultiscaleNetwork(spikenet, oscnet)

    multinet.add_transducers(TEST_TRANSDUCERS)


def test_build_projections():

    # Builds the spiking network
    spikenet = SpikingNetwork.from_yaml(TEST_NET, TEST_NEURONS)
    spikenet.build()

    # Builds the oscillator network
    oscnet = OscillatorNetwork.homogeneous_from_TVB('brain_data/connectivity_76.zip', 
                                             {'oscillator_type':'jansen-rit'}, 
                                             global_weight=0.1, 
                                             conduction_speed=1)
    oscnet.build()

    # states = np.random.uniform(0, 0.05, size=6*oscnet.n_oscillators).reshape(oscnet.n_oscillators, 6)
    # oscnet.init(states, dt=1)

    multinet = MultiscaleNetwork(spikenet, oscnet)

    multinet.add_transducers(TEST_TRANSDUCERS)
    np.random.seed(1997)
    T2O_delays = np.random.uniform(10, 11, size = (multinet.n_transducers, multinet.n_oscillators)).astype(np.float32)
    T2O_weights = np.random.uniform(0, 1, size = (multinet.n_transducers, multinet.n_oscillators)).astype(np.float32)

    # Deletes some links
    T2O_weights[(T2O_weights < 0.5)] = 0.0

    O2T_delays = np.random.uniform(10, 11, size=(multinet.n_oscillators, multinet.n_transducers)).astype(np.float32)
    O2T_weights = np.random.uniform(0, 1, size=(multinet.n_oscillators, multinet.n_transducers)).astype(np.float32)
    # Deletes some links
    O2T_weights[(O2T_weights < 0.5)] = 0.0

    T2Oproj = Projection(T2O_weights, T2O_delays)
    O2Tproj = Projection(O2T_weights, O2T_delays)

    multinet.build_multiscale_projections(T2O=T2Oproj, O2T=O2Tproj)

    states = np.random.uniform(0, 0.05, size=6*oscnet.n_oscillators).reshape(oscnet.n_oscillators, 6)
    oscnet.initialize(states, dt=1)


def test_initialize():
    # Builds the spiking network
    spikenet = SpikingNetwork.from_yaml(TEST_NET, TEST_NEURONS)
    spikenet.build()

    # Builds the oscillator network
    oscnet = OscillatorNetwork.homogeneous_from_TVB('brain_data/connectivity_76.zip', 
                                             {'oscillator_type':'jansen-rit'}, 
                                             global_weight=0.1, 
                                             conduction_speed=1)
    oscnet.build()

    multinet = MultiscaleNetwork(spikenet, oscnet)

    multinet.add_transducers(TEST_TRANSDUCERS)
    np.random.seed(1997)
    T2O_delays = np.random.uniform(10, 11, size = (multinet.n_transducers, multinet.n_oscillators)).astype(np.float32)
    T2O_weights = np.random.uniform(0, 1, size = (multinet.n_transducers, multinet.n_oscillators)).astype(np.float32)

    # Deletes some links
    T2O_weights[(T2O_weights < 0.5)] = 0.0

    O2T_delays = np.random.uniform(10, 11, size=(multinet.n_oscillators, multinet.n_transducers)).astype(np.float32)
    O2T_weights = np.random.uniform(0, 1, size=(multinet.n_oscillators, multinet.n_transducers)).astype(np.float32)
    # Deletes some links
    O2T_weights[(O2T_weights < 0.5)] = 0.0

    T2Oproj = Projection(T2O_weights, T2O_delays)
    O2Tproj = Projection(O2T_weights, O2T_delays)

    multinet.build_multiscale_projections(T2O=T2Oproj, O2T=O2Tproj)

    states = np.random.uniform(0, 0.05, size=6*oscnet.n_oscillators).reshape(oscnet.n_oscillators, 6)
    
    multinet.set_evolution_contextes(dt_short=0.1, dt_long=1.0)
    multinet.initialize(states)

def test_run():
    # Builds the spiking network
    spikenet = SpikingNetwork.from_yaml(TEST_NET, TEST_NEURONS)
    spikenet.build()

    # Builds the oscillator network
    oscnet = OscillatorNetwork.homogeneous_from_TVB('brain_data/connectivity_76.zip', 
                                             {'oscillator_type':'jansen-rit'}, 
                                             global_weight=0.1, 
                                             conduction_speed=1)
    oscnet.build()

    multinet = MultiscaleNetwork(spikenet, oscnet)

    multinet.add_transducers(TEST_TRANSDUCERS)
    np.random.seed(1997)
    T2O_delays = np.random.uniform(10, 11, size = (multinet.n_transducers, multinet.n_oscillators)).astype(np.float32)
    T2O_weights = np.random.uniform(0, 1, size = (multinet.n_transducers, multinet.n_oscillators)).astype(np.float32)

    # Deletes some links
    T2O_weights[(T2O_weights < 0.5)] = 0.0

    O2T_delays = np.random.uniform(10, 11, size=(multinet.n_oscillators, multinet.n_transducers)).astype(np.float32)
    O2T_weights = np.random.uniform(0, 1, size=(multinet.n_oscillators, multinet.n_transducers)).astype(np.float32)
    # Deletes some links
    O2T_weights[(O2T_weights < 0.5)] = 0.0

    T2Oproj = Projection(T2O_weights, T2O_delays)
    O2Tproj = Projection(O2T_weights, O2T_delays)

    multinet.build_multiscale_projections(T2O=T2Oproj, O2T=O2Tproj)

    states = np.random.uniform(0, 0.05, size=6*oscnet.n_oscillators).reshape(oscnet.n_oscillators, 6)
    
    multinet.set_evolution_contextes(dt_short=0.1, dt_long=1.0)
    multinet.initialize(states)

    print("Printing max times:")
    for _ in [O2T_delays, T2O_delays, oscnet.features['connectivity']['delays']]:
        print(np.min(_[~np.isnan(_)]))

    multinet.run(time=1000)

def test_get_history():
    # Builds the spiking network
    spikenet = SpikingNetwork.from_yaml(TEST_NET, TEST_NEURONS)
    spikenet.monitorize_spikes()
    # spikenet.monitorize_states("Albert")
    spikenet.build()

    # Builds the oscillator network
    oscnet = OscillatorNetwork.homogeneous_from_TVB('brain_data/connectivity_76.zip', 
                                             {'oscillator_type':'jansen-rit'}, 
                                             global_weight=0.1, 
                                             conduction_speed=1)
    oscnet.build()

    multinet = MultiscaleNetwork(spikenet, oscnet)

    multinet.add_transducers(TEST_TRANSDUCERS)
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

    states = np.random.uniform(0,20 , size=6*oscnet.n_oscillators).reshape(oscnet.n_oscillators, 6)
    
    multinet.set_evolution_contextes(dt_short=0.1, dt_long=1.0)
    multinet.initialize(states)

    print("Printing max times:")
    for _ in [O2T_delays, T2O_delays, oscnet.features['connectivity']['delays']]:
        print(np.min(_[~np.isnan(_)]))

    T_sec = 2
    multinet.run(time=T_sec*1000)

    pops = multinet.spiking_network.populations
    oscs = multinet.oscillator_network.oscillators

    # print(pops)
    # print(oscs)

    import matplotlib.pyplot as plt
    N_osc_plot = 5
    fig, axes = plt.subplots(len(pops.keys())+ N_osc_plot, 1, sharex=True)

    for i in range(len(pops.keys())):
        axes[i].set_ylabel(list(pops.keys())[i])
        spikes = bin_spikes(pops[list(pops.keys())[i]].get_data('spikes'))
        time = np.linspace(0, T_sec, len(spikes))
        mask = time > 0.5
        axes[i].plot(time[mask], spikes[mask])

    for i in range(len(pops), N_osc_plot + len(pops)):
        axes[i].set_ylabel(list(oscs.keys())[i])
        signal = oscs[list(oscs.keys())[i]].history[:, 0]
        time = np.linspace(0, T_sec, len(signal))
        mask = time > 0.5
        axes[i].plot(time[mask], signal[mask])

    plt.autoscale(enable=True, axis='y')
    # plt.xlim(500, 1000)
    plt.show()

    # plt.plot(multinet.spiking_network.populations["Albert"].get_data("states")[:, :5, 0])
    # plt.show()

if __name__ == "__main__":
    test_transducers_from_list()