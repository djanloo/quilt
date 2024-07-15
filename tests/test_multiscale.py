from quilt.builder import MultiscaleNetwork, SpikingNetwork, OscillatorNetwork
from quilt.interface.base import Projection
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
                                weight=0.1)
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

if __name__ == "__main__":
    test_run()