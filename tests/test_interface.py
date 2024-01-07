from quilt.builder import SpikingNetwork

TEST_NET="tests/test_spiking.yaml"
TEST_NEURONS = "tests/test_neurons.yaml"

def test_yaml_builder():
    SpikingNetwork.from_yaml(TEST_NET)

def test_run():
    spikenet = SpikingNetwork.from_yaml(TEST_NET)
    spikenet.interface.run(dt=0.1, time=10)

def test_neuorons():
    spikenet = SpikingNetwork.from_yaml(TEST_NEURONS)
    spikenet.interface.run(dt=0.1, time=10)

"""
Input/Output devices
"""
def test_spike_monitor():
    spikenet = SpikingNetwork.from_yaml(TEST_NET)
    spikenet.populations['Albert'].monitorize_spikes()

def test_state_monitor():
    spikenet = SpikingNetwork.from_yaml(TEST_NET)
    spikenet.populations['Albert'].monitorize_states()

def test_injector():
    spikenet = SpikingNetwork.from_yaml(TEST_NET)
    spikenet.populations["Albert"].add_injector(0.5, 0.0, 2)

if __name__=="__main__":
    
    import numpy as np
    import matplotlib.pyplot as plt 

    spikenet = SpikingNetwork.from_yaml("spiking.yaml")
    spikenet.populations["Albert"].add_injector(300, 0, 20.0)
    spikenet.populations["Albert"].monitorize_spikes()
    spikenet.populations["Albert"].monitorize_states()
    spikenet.populations["MonaLisa"].monitorize_spikes()
    spikenet.populations["MonaLisa"].monitorize_states()

    spikenet.interface.run(dt=0.1, time=300)

    albert_spikes = spikenet.populations['Albert'].get_data()["spikes"]
    monalisa_spikes = spikenet.populations['MonaLisa'].get_data()["spikes"]
    plt.step(np.arange(len(albert_spikes)), albert_spikes, alpha=0.3)
    plt.step(np.arange(len(albert_spikes)), monalisa_spikes, alpha=0.3)
    N = 10
    plt.plot(np.convolve(albert_spikes, np.ones(N)/N))

    states = np.array(spikenet.populations['Albert'].get_data()['states'])
    # print(states.shape)
    plt.figure(2)
    for i in range(1):
        plt.plot(states[:, i, 0],marker=".")
        # print(states[:, i, 0])
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
