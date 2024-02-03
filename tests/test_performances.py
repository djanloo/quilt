from quilt.builder import SpikingNetwork, NeuronCatalogue

nn = NeuronCatalogue.from_yaml("bg_analysis/ortone_neurons.yaml")
sn = SpikingNetwork.from_yaml("bg_analysis/ortone_network.yaml", nn)

sn.build()
sn.run(dt=0.1, time=600)