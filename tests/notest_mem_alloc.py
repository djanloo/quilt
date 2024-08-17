from quilt.builder import OscillatorNetwork
import numpy as np
from quilt.interface.base import set_verbosity

set_verbosity(4)


for i in range(10):
    net = OscillatorNetwork.homogeneous_from_TVB('tests/connectivity_76.zip', 
                                                {'oscillator_type': 'jansen-rit'},
                                                global_weight=1, 
                                                conduction_speed=1.0)

    net.build()
    states = np.random.uniform(0, 0.2 , size=6*net.n_oscillators).reshape(net.n_oscillators, 6)
    net.initialize(states, dt=1)

    net.run(20)