from quilt.builder import SpikingNetwork, NeuronCatalogue

TEST_NET="tests/test_spiking.yaml"
TEST_NEURONS = "tests/test_neurons.yaml"
TEST_PARAMS = "tests/test_params.yaml"

"""Parameters and models"""
def test_paramaps():
    import quilt.bin.neur as neur
    paramap = neur.ParaMap(dict(neuron_type='aeif', E_rest=-70, 
                                E_thr=0, tau_m=10, E_reset=-65,E_exc=0, E_inh=-70,
                                C_m=40, tau_e=10, tau_i=12, tau_refrac=0,
                                Delta=3, exp_threshold=-30, 
                                ada_a=1, ada_b=3, ada_tau_w=4))
    sn = neur.SpikingNetwork("a")
    pop = neur.Population(10, paramap, sn)

def test_catalogue():
    catalogue = NeuronCatalogue.from_yaml(TEST_PARAMS)

def test_yaml_builder():
    test_catalogue = NeuronCatalogue.from_yaml(TEST_PARAMS)
    SpikingNetwork.from_yaml(TEST_NET, test_catalogue)

def test_run():
    test_catalogue = NeuronCatalogue.from_yaml(TEST_PARAMS)
    spikenet = SpikingNetwork.from_yaml(TEST_NET, test_catalogue)
    spikenet.run(dt=0.1, time=10)

def test_neurons():
    test_catalogue = NeuronCatalogue.from_yaml(TEST_PARAMS)
    spikenet = SpikingNetwork.from_yaml(TEST_NEURONS, test_catalogue)
    spikenet.run(dt=0.1, time=10)

def test_delay_control():
    test_catalogue = NeuronCatalogue.from_yaml(TEST_PARAMS)
    spikenet = SpikingNetwork.from_yaml(TEST_NET, test_catalogue)
    try:
        spikenet.run(dt=2.0, time=10)
    except RuntimeError:
        print("This thest successfully failed.")
    else:
        raise RuntimeError("Synaptic delay test check did not fail, but it was supposed to")


"""
Input/Output devices
"""
def test_spike_monitor():
    test_catalogue = NeuronCatalogue.from_yaml(TEST_PARAMS)
    spikenet = SpikingNetwork.from_yaml(TEST_NET, test_catalogue)
    spikenet.populations['Albert'].monitorize_spikes()

def test_state_monitor():
    test_catalogue = NeuronCatalogue.from_yaml(TEST_PARAMS)
    spikenet = SpikingNetwork.from_yaml(TEST_NET, test_catalogue)
    spikenet.populations['Albert'].monitorize_states()

def test_injector():
    test_catalogue = NeuronCatalogue.from_yaml(TEST_PARAMS)
    spikenet = SpikingNetwork.from_yaml(TEST_NET, test_catalogue)
    spikenet.populations["Albert"].add_injector(0.5, 0.0, 2)

if __name__=="__main__":
    # test_yaml_builder()
    import numpy as np
    import matplotlib.pyplot as plt 

    test_catalogue = NeuronCatalogue.from_yaml(TEST_PARAMS)
    spikenet = SpikingNetwork.from_yaml("tests/spiking.yaml", test_catalogue)
    spikenet.populations["Albert"].add_injector(300, 0, 10.0)
    spikenet.populations["Albert"].monitorize_spikes()
    spikenet.populations["Albert"].monitorize_states()
    spikenet.populations["MonaLisa"].monitorize_spikes()
    spikenet.populations["MonaLisa"].monitorize_states()

    spikenet.interface.run(dt=0.1, time=200)

    albert_spikes = spikenet.populations['Albert'].get_data()["spikes"]
    monalisa_spikes = spikenet.populations['MonaLisa'].get_data()["spikes"]
    plt.step(np.arange(len(albert_spikes)), albert_spikes, alpha=0.3)
    plt.step(np.arange(len(albert_spikes)), monalisa_spikes, alpha=0.3)
    N = 10
    plt.plot(np.convolve(albert_spikes, np.ones(N)/N))

    states = np.array(spikenet.populations['Albert'].get_data()['states'])

    plt.figure(2)
    for i in range(1):
        plt.plot(states[:, i, 0],marker=".")
        plt.title("V")

    plt.figure(3)
    for i in range(5):
        plt.plot(states[:, i, 1],marker=".")
        plt.title("gsyn_exc")

    plt.figure(4)
    for i in range(5):
        plt.plot(states[:, i, 2],marker=".")
        plt.title("gsyn_inh")

    plt.figure(5)
    for i in range(5):
        plt.plot(states[:, i, 3],marker=".")
        plt.title("u")

    states = np.array(spikenet.populations['MonaLisa'].get_data()['states'])

    plt.figure(6)
    for idx, feat in enumerate(["V", "gsyne", "gsyni", "u"]):
        plt.plot(states[:, 0, idx], label=feat)
    plt.title("MonaLisa")
    plt.legend()


    plt.show()
