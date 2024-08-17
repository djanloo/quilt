from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

cimport cinterface as cinter
cimport base

VERBOSITY = 1

def set_verbosity(v):
    global VERBOSITY
    VERBOSITY = v

cdef class Oscillator:

    def __cinit__(self):
        pass

    cdef wrap(self, cinter.Oscillator * osc):
        self._osc = osc

    @property
    def history(self):
        return np.array(self._osc.get_history())

    @property
    def rate_history(self):
        return np.array(self._osc.get_rate_history())

    @property
    def input_history(self):
        return np.array(self._osc.input_history)
    
    @property
    def eeg(self):
        return np.array(self._osc.get_eeg())

    def print_info(self):
        self._osc.print_info()


cdef class OscillatorNetwork:

    # cdef cinter.OscillatorNetwork * _oscillator_network
    # cdef cinter.EvolutionContext * _evo
    # cdef public list oscillators

    # This is needed otherwise OscillatorNetwork() will take
    # whatever number of arguments
    def __init__(self):
        self.oscillators = list()
        
    @classmethod
    def homogeneous(cls, int N, base.ParaMap params):
        cdef OscillatorNetwork net = cls()
        net._oscillator_network = new cinter.OscillatorNetwork(N, params._paramap)
        
        # this is for bug #37
        # stores a copy of the paramap to prevent early deallocation
        net._paramap = params

        # wraps the oscillators
        cdef Oscillator temp
        for i in range(net._oscillator_network.oscillators.size()):
            temp = Oscillator()
            temp.wrap(net._oscillator_network.oscillators[i])
            net.oscillators.append(temp)

        return net
    
    @classmethod
    def inhomogeneous(cls, list params):

        cdef base.ParaMapList paramap_list = base.ParaMapList()
        paramap_list.load_list(params)

        cdef OscillatorNetwork net = cls()
        net._oscillator_network = new cinter.OscillatorNetwork(paramap_list.paramap_vector)

        # wraps the oscillators
        cdef Oscillator temp
        for i in range(net._oscillator_network.oscillators.size()):
            temp = Oscillator()
            temp.wrap(net._oscillator_network.oscillators[i])
            net.oscillators.append(temp)
            
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
        print("Cython: called dealloc on OscillatorNetwork")
        if self._oscillator_network != NULL:
            del self._oscillator_network

