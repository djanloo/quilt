from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

import numpy as np
cimport numpy as np

cimport quilt.interface.cinterface as cinter
cimport quilt.interface.base as base

VERBOSITY = 1

def set_verbosity(v):
    global VERBOSITY
    VERBOSITY = v

cdef class Oscillator:
    # cdef shared_ptr[cinter.Oscillator] _osc

    def __cinit__(self):
        pass

    cdef wrap(self, shared_ptr[cinter.Oscillator] osc):
        self._osc = osc

    @property
    def history(self):
        # Gets the object from the shared pointer
        cdef cinter.Oscillator * osc = self._osc.get()
        return np.array(osc.get_history())
    
    @property
    def eeg(self):
        # Gets the object from the shared pointer
        cdef cinter.Oscillator * osc = self._osc.get()
        return np.array(osc.get_eeg())


cdef class OscillatorNetwork:

    # cdef cinter.OscillatorNetwork * _oscillator_network
    # cdef cinter.EvolutionContext * _evo
    # cdef public list oscillators

    # This is needed otherwise OscillatorNetwork() will take
    # whatever number of arguments
    def __init__(self):
        self.oscillators = list()

    def wrap_oscillators(self):
        cdef unsigned int i = 0
        cdef shared_ptr[cinter.Oscillator] _osc
        cdef Oscillator osc

        for i in range(self._oscillator_network.oscillators.size()):
            _osc = self._oscillator_network.oscillators[i]
            osc = Oscillator()
            osc.wrap(_osc)
            self.oscillators.append(osc)
        
    @classmethod
    def homogeneous(cls, int N, base.ParaMap params):
        cdef OscillatorNetwork net = cls()
        net._oscillator_network = new cinter.OscillatorNetwork(N, params._paramap)
        net.wrap_oscillators()
        return net
    
    @classmethod
    def inhomogeneous(cls, list params):

        cdef base.ParaMapList paramap_list = base.ParaMapList()
        paramap_list.load_list(params)

        cdef OscillatorNetwork net = cls()
        net._oscillator_network = new cinter.OscillatorNetwork(paramap_list.paramap_vector)
        net.wrap_oscillators()
        return net

    def build_connections(self, base.Projection proj, base.ParaMap params):
        if self._oscillator_network != NULL:
            self._oscillator_network.build_connections(proj._projection , params._paramap)
        else:
            raise RuntimeError("Oscillator network does not have oscillators to link yet.")

    def run(self, time=1):
        self._oscillator_network.run(self._evo, time, VERBOSITY)
    
    def initialize(self, np.ndarray[np.double_t, ndim=2, mode='c'] states, dt=1.0):
        self._evo = new cinter.EvolutionContext(dt)
        self._oscillator_network.initialize(self._evo, states)

    def __dealloc__(self):
        if self._oscillator_network != NULL:
            del self._oscillator_network

