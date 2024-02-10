from libcpp cimport string

from time import time
from libc.stdlib cimport malloc
from libcpp.vector cimport vector

import ctypes

import numpy as np
cimport numpy as np

cimport quilt.interface.cinterface as cinter
cimport quilt.interface.base_objects as base_objects

get_class = {"harmonic": harmonic_oscillator} 

cdef class OscillatorNetwork:

    cdef cinter.OscillatorNetwork * _oscillator_network
    cdef cinter.EvolutionContext * _evo

    def __cinit__(self):
        self._oscillator_network = new cinter.OscillatorNetwork()

    def build_connections(self, base_objects.Projection proj):
        cdef int i,j
        if proj.start_dimension != proj.end_dimension:
            raise ValueError("Dimension mismatch in oscillator projection")
        for i in range(proj.start_dimension):
            for j in range(proj.end_dimension):
                self._oscillator_network.oscillators[i].connect(self._oscillator_network.oscillators[j], 
                                                                proj.weights[i,j], proj.delays[i,j])

    def run(self, dt=0.1, time=1):
        self._evo = new cinter.EvolutionContext(dt)
        self._oscillator_network.run(self._evo, time)


cdef class harmonic_oscillator:
    cdef:
        cinter.harmonic_oscillator * _oscillator
        OscillatorNetwork osc_net
        base_objects.ParaMap paramap

    def __cinit__(self, dict params, OscillatorNetwork oscillator_network):
        """Do the checks here instead C++?"""
        params['oscillator_type'] = 'harmonic'
        self.paramap = base_objects.ParaMap(params)
        self._oscillator = <cinter.harmonic_oscillator *> new cinter.harmonic_oscillator(self.paramap._paramap, oscillator_network._oscillator_network)
        
    @property
    def history(self):
        return np.array(self._oscillator.get_history())
    


