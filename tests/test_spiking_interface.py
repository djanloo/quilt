from quilt.builder import SpikingNetwork, NeuronCatalogue

TEST_NET="tests/test_spiking.yaml"
TEST_EVOLUTION = "tests/test_evolution.yaml"
TEST_NEURONS    = "tests/test_neurons.yaml"
TEST_SUSCEPTIBILITY_1 = "tests/test_susceptibility.yaml"
TEST_SUSCEPTIBILITY_2 = "tests/test_susceptibility_2.yaml"

"""Parameters and models"""
def test_paramaps():
    import quilt.interface.base_objects as base_objs
    import quilt.interface.spiking as spiking
    paramap = base_objs.ParaMap(dict(neuron_type='aeif', E_l=-70, 
                                V_peak=0, G_L=10, V_reset=-65,E_ex=0, E_in=-70,
                                C_m=40, tau_ex=10, tau_in=12, tau_refrac=0,
                                delta_T=3, V_th=-30, 
                                ada_a=1, ada_b=3, ada_tau_w=4))
    sn = spiking.SpikingNetwork("a")
    pop = spiking.Population(10, paramap, sn)

def test_catalogue():
    catalogue = NeuronCatalogue.from_yaml(TEST_NEURONS)

def test_yaml_builder():
    SpikingNetwork.from_yaml(TEST_NET, TEST_NEURONS)

def test_run():
    spikenet = SpikingNetwork.from_yaml(TEST_NET, TEST_NEURONS)
    spikenet.run(dt=0.1, time=3)

# def test_neurons():
#     spikenet = SpikingNetwork.from_yaml(TEST_NEURONS, TEST_PARAMS)
#     spikenet.run(dt=0.1, time=3)

# def test_delay_control():
#     test_catalogue = NeuronCatalogue.from_yaml(TEST_PARAMS)
#     spikenet = SpikingNetwork.from_yaml(TEST_NET, test_catalogue)
#     try:
#         spikenet.run(dt=2.0, time=3)
#     except RuntimeError:
#         print("This thest successfully failed.")
#     else:
#         raise RuntimeError("Synaptic delay test check did not fail, but it was supposed to")


"""
Input/Output devices
"""
def test_spike_monitor():
    spikenet = SpikingNetwork.from_yaml(TEST_NET, TEST_NEURONS)
    spikenet.build()
    spikenet.populations['Albert'].monitorize_spikes()

def test_state_monitor():
    spikenet = SpikingNetwork.from_yaml(TEST_NET, TEST_NEURONS)
    spikenet.build()
    spikenet.populations['Albert'].monitorize_states()
    spikenet.run(dt=0.1, time=1)

def test_const_curr_injector():
    spikenet = SpikingNetwork.from_yaml(TEST_NET, TEST_NEURONS)
    spikenet.build()
    spikenet.populations["Albert"].add_const_curr_injector(0.5, 0.0, 2)
    spikenet.run(dt=0.1, time=1)

def test_poisson_spike_injector():
    spikenet = SpikingNetwork.from_yaml(TEST_NET, TEST_NEURONS)
    spikenet.build()
    spikenet.populations["Albert"].add_poisson_spike_injector(250, 0.3, 0.05)
    spikenet.run(dt=0.1, time=1)

def test_parametric_suscettivity():
    from quilt.builder import ParametricSpikingNetwork
    sn = ParametricSpikingNetwork.from_yaml(TEST_NET, TEST_NEURONS, TEST_SUSCEPTIBILITY_1)
    sn.set_parameters(alpha=0.1)
    sn.build()

def test_parametric_set():
    from quilt.builder import ParametricSpikingNetwork
    sn = ParametricSpikingNetwork.from_yaml(TEST_NET, TEST_NEURONS, TEST_SUSCEPTIBILITY_2)
    sn.set_parameters(gamma=-6)
    sn.build()

def test_multiparametric_sn():
    from quilt.builder import ParametricSpikingNetwork
    sn = ParametricSpikingNetwork.from_yaml(TEST_NET, TEST_NEURONS, [TEST_SUSCEPTIBILITY_1, TEST_SUSCEPTIBILITY_2])
    sn.set_parameters(alpha=0.1, gamma=-10)
    sn.build()

if __name__=="__main__":
    test_parametric_set()
    exit()
    