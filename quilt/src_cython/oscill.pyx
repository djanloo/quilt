cimport quilt.src_cython.cinterface as cinter

# from time import time
# from libc.stdlib cimport malloc
# from libcpp.vector cimport vector

# import ctypes

# import numpy as np
# cimport numpy as np


cdef class Oscillator:
    cdef cinter.Oscillator * _oscillator
    cdef cinter.EvolutionContext * _evo

    def __cinit__(self, k, x, v):
        self._oscillator = new cinter.Oscillator()
        
    @property
    def history(self):
        return self._oscillator.history

cdef class dummy_osc(Oscillator):

    def __cinit__(self, k, x, v):
        self._oscillator = new cinter.dummy_osc(k, x, v)
        
    @property
    def history(self):
        return self._oscillator.history

cdef class OscillatorNetwork:
    cdef cinter.OscillatorNetwork * _osc_net
    def __cinit__(self):
        self._osc_net = new cinter.OscillatorNetwork()
    
    def add_oscillator(self, Oscillator oscillator):
        self._osc_net.add_oscillator(oscillator._oscillator)
    
    def run(self, dt=0.1, t=1):
        self._evo = new cinter.EvolutionContext(dt)
        self._osc_net.run(self._evo, t)

