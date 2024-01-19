from quilt.builder import SpikingNetwork, NeuronCatalogue

nn = NeuronCatalogue.from_yaml("tests/basal_ganglia_neurons.yaml")
sn = SpikingNetwork.from_yaml("tests/basal_ganglia_network.yaml", nn)

sn.build()

for pop in sn.populations.values():
    pop.add_poisson_spike_injector(1000, 0.5, t_min=0, t_max=-1)

sn.run(dt=0.1, time=600)