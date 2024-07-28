from quilt.builder import SpikingNetwork, NeuronCatalogue
from quilt.interface.base import set_verbosity

set_verbosity(2)

sn = SpikingNetwork.from_yaml("bg_network_dispersive.yaml", "bg_neurons.yaml")

sn.monitorize_spikes()
sn.build()
sn.run(dt=0.1, time=600)


import matplotlib.pyplot as plt

plt.plot(sn.populations["D1"].get_data('spikes'))
plt.show()