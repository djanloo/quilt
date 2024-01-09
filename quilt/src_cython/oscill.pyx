cimport quilt.src_cython.cinterface as cinter

# from time import time
# from libc.stdlib cimport malloc
# from libcpp.vector cimport vector

# import ctypes

# import numpy as np
# cimport numpy as np

cdef class dummy_osc:
    cdef cinter.dummy_osc * _dummy_osc
    cdef cinter.EvolutionContext * _evo

    def __cinit__(self, k, x, v):
        self._dummy_osc = new cinter.dummy_osc(k, x, v)
        
    @property
    def history(self):
        return self._dummy_osc.history
    
    def run(self,dt=0.1, t=1):
        self._evo = new cinter.EvolutionContext(dt)
        self._dummy_osc.run()

cdef class OscillatorNetwork:
    cdef 
    def __cinit__(self)
