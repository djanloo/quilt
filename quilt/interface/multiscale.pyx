cimport cinterface as cinter
cimport base

cimport spiking
cimport oscill

import numpy as np
cimport numpy as np
# from libcpp.memory cimport shared_ptr

cdef class Transducer:
    cdef cinter.Transducer * _transducer

    def __cinit__(self, spiking.Population population, base.ParaMap params, MultiscaleNetwork multinet):
        self._transducer = new cinter.Transducer(population._population, params._paramap, multinet._multiscale_network)

cdef class MultiscaleNetwork:
    cdef cinter.MultiscaleNetwork * _multiscale_network 
    cdef cinter.EvolutionContext * _evo_short
    cdef cinter.EvolutionContext * _evo_long

    # def __cinit__(self, spiking.SpikingNetwork spikenet, oscill.OscillatorNetwork oscnet):
    #     self._multiscale_network = new cinter.MultiscaleNetwork(spikenet._spiking_network, oscnet._oscillator_network)

    def __init__(self, spiking.SpikingNetwork spikenet, oscill.OscillatorNetwork oscnet):
        self._multiscale_network = new cinter.MultiscaleNetwork(spikenet._spiking_network, oscnet._oscillator_network)

    def add_transducer(self, spiking.Population population, base.ParaMap params):
        self._multiscale_network.add_transducer(population._population, params._paramap)

    def build_multiscale_projections(self,  base.Projection T2O, base.Projection O2T):
        self._multiscale_network.build_multiscale_projections(T2O._projection, O2T._projection)

    def set_evolution_contextes(self, dt_short=0.1, dt_long=1.0):
        self._evo_short = new cinter.EvolutionContext(dt_short)
        self._evo_long = new cinter.EvolutionContext(dt_long)

        self._multiscale_network.set_evolution_contextes(self._evo_short, self._evo_long)
    
    def initialize(self, np.ndarray[np.double_t, ndim=2, mode='c'] states):
        self._multiscale_network.oscnet.initialize(self._evo_long, states)
        self._multiscale_network.spikenet.run(self._evo_short, self._evo_long.now, 1)

    def run(self, time=10):
        self._multiscale_network.run(time, 1)
