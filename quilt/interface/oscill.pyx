from libcpp.vector cimport vector
import warnings

import numpy as np
cimport numpy as np

cimport quilt.interface.cinterface as cinter
cimport quilt.interface.base as base

VERBOSITY = 1

def set_verbosity(v):
    global VERBOSITY
    VERBOSITY = v

cdef class OscillatorNetwork:

    cdef cinter.OscillatorNetwork * _oscillator_network
    cdef cinter.EvolutionContext * _evo

    def __cinit__(self, N, base.ParaMap params):
        self._oscillator_network = new cinter.OscillatorNetwork(N, params._paramap)

    def build_connections(self, base.Projection proj, base.ParaMap params):
        self._oscillator_network.build_connections(proj._projection , params._paramap)

    def run(self, time=1):
        self._oscillator_network.run(self._evo, time, VERBOSITY)
    
    def init(self, np.ndarray[np.double_t, ndim=2, mode='c'] states, dt=1.0):
        self._evo = new cinter.EvolutionContext(dt)
        self._oscillator_network.initialize(self._evo, states)

    def __dealloc__(self):
        if self._oscillator_network != NULL:
            del self._oscillator_network

