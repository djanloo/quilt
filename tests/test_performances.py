from quilt.builder import SpikingNetwork, NeuronCatalogue

sn = SpikingNetwork.from_yaml("bg_analysis/ortone_network_dispersive.yaml", "bg_analysis/ortone_neurons.yaml")

sn.monitorize_spikes()
sn.build()
sn.run(dt=0.1, time=60)

import matplotlib.pyplot as plt

plt.plot(sn.populations["D1"].get_data('spikes'))
plt.show()