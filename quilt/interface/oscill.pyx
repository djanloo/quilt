cimport quilt.interface.cinterface as cinter
from libcpp cimport string

OSCILLATOR_TYPES = {"harmonic":0}

# from time import time
# from libc.stdlib cimport malloc
# from libcpp.vector cimport vector

# import ctypes

# import numpy as np
# cimport numpy as np


# cdef class Oscillator:
#     cdef cinter.Oscillator * _oscillator
#     cdef cinter.EvolutionContext * _evo

#     def __cinit__(self, k, x, v):
#         self._oscillator = new cinter.Oscillator()
        
#     @property
#     def history(self):
#         return self._oscillator.history

# cdef class dummy_osc(Oscillator):

#     def __cinit__(self, k, x, v):
#         self._oscillator = new cinter.dummy_osc(k, x, v)
        
#     @property
#     def history(self):
#         return self._oscillator.history

cdef class OscillatorNetwork:
    cdef cinter.OscillatorNetwork * _osc_net
    cdef cinter.EvolutionContext * evo

    def __cinit__(self, str oscillator, cinter.ParaMap params):
        try:
            oscillator_type = OSCILLATOR_TYPES[oscillator]
        except KeyError:
            raise KeyError(f"Oscillator '{oscillator}' is not in {list(OSCILLATOR_TYPES.keys())}")
        self._osc_net = new cinter.OscillatorNetwork(oscillator_type, )

    def run(self, dt=0.1, t=1):
        self._evo = new cinter.EvolutionContext(dt)
        self._osc_net.run(self._evo, t)

