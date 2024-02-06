from quilt.builder import SpikingNetwork, NeuronCatalogue

sn = SpikingNetwork.from_yaml("bg_analysis/ortone_network_dispersive.yaml", "bg_analysis/ortone_neurons.yaml")

sn.build()
sn.run(dt=0.1, time=600)