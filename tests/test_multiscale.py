from quilt.builder import MultiscaleNetwork, SpikingNetwork, OscillatorNetwork

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



if __name__ == "__main__":
    test_transducers_from_list()