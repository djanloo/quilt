from quilt.builder import SpikingNetwork, NeuronCatalogue

nn = NeuronCatalogue.from_yaml("tests/basal_ganglia_neurons.yaml")
sn = SpikingNetwork.from_yaml("tests/basal_ganglia_network.yaml", nn)

sn.run(dt=0.1, time=500)