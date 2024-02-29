from libcpp.vector cimport vector
import warnings

import numpy as np
cimport numpy as np

cimport quilt.interface.cinterface as cinter
cimport quilt.interface.base as base

get_class = {"harmonic": harmonic_oscillator, "jansen-rit": jansen_rit_oscillator} 

VERBOSITY = 1

def set_verbosity(v):
    global VERBOSITY
    VERBOSITY = v

cdef class OscillatorNetwork:

    cdef cinter.OscillatorNetwork * _oscillator_network
    cdef cinter.EvolutionContext * _evo

    def __cinit__(self):
        self._oscillator_network = new cinter.OscillatorNetwork()

    def build_connections(self, base.Projection proj):
        cdef int i,j
        some_delay_is_zero = False
        if proj.start_dimension != proj.end_dimension:
            raise ValueError("Dimension mismatch in oscillator projection")
        for i in range(proj.start_dimension):
            for j in range(proj.end_dimension):
                if proj.weights[i, j] != 0 :
                    if proj.delays[i, j] == 0:
                        some_delay_is_zero = True
                    self._oscillator_network.oscillators[i].connect(self._oscillator_network.oscillators[j], 
                                                                    proj.weights[i,j], proj.delays[i,j])

        if some_delay_is_zero:
            warnings.warn("Some delay in the network is zero. This can lead to undefinite behaviours (for now)")
    
    def run(self, time=1):
        self._oscillator_network.run(self._evo, time, VERBOSITY)
    
    def init(self, np.ndarray[np.double_t, ndim=2, mode='c'] states, dt=1.0):
        self._evo = new cinter.EvolutionContext(dt)

        n_oscillators = states.shape[0]
        space_dimension = states.shape[1]

        cdef vector[vector[double]] _states
        _states = vector[vector[double]](n_oscillators)

        for i in range(n_oscillators):
            row = vector[double](space_dimension)
            for j in range(space_dimension):
                row[j] = states[i,j]

            _states[i] = row
        
        self._oscillator_network.initialize(self._evo, _states)


cdef class harmonic_oscillator:
    cdef:
        cinter.harmonic_oscillator * _oscillator
        OscillatorNetwork osc_net
        base.ParaMap paramap

    def __cinit__(self, dict params, OscillatorNetwork oscillator_network):
        params['oscillator_type'] = 'harmonic'
        self.paramap = base.ParaMap(params)
        self._oscillator = <cinter.harmonic_oscillator *> new cinter.harmonic_oscillator(self.paramap._paramap, oscillator_network._oscillator_network)
        
    @property
    def history(self):
        return np.array(self._oscillator.get_history())
    

cdef class jansen_rit_oscillator:
    cdef:
        cinter.jansen_rit_oscillator * _oscillator
        OscillatorNetwork osc_net
        base.ParaMap paramap

    def __cinit__(self, dict params, OscillatorNetwork oscillator_network):
        params['oscillator_type'] = 'jansen-rit'
        self.paramap = base.ParaMap(params)
        self._oscillator = <cinter.jansen_rit_oscillator *> new cinter.jansen_rit_oscillator(self.paramap._paramap, oscillator_network._oscillator_network)
        
    @property
    def history(self):
        return np.array(self._oscillator.get_history())