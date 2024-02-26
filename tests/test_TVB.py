import numpy as np
from quilt.builder import OscillatorNetwork
import matplotlib.pyplot as plt

net = OscillatorNetwork.homogeneous_from_TVB('connectivity_66.zip', {'oscillator_type':'jansen-rit'})
net.build()

states = np.random.uniform(0, 0.05, size=6*net.n_oscillators).reshape(net.n_oscillators, 6)
net.init(states, dt=1)
net.run(time=10000, dt=1)

for name, number in zip(net.oscillators.keys(), range(2)):
    plt.plot(net.oscillators[name].history[:, 0], label=name)
plt.xlabel("t [ms]")
plt.legend()
plt.show()