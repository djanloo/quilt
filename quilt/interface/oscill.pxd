cimport cinterface as cinter

from base cimport ParaMap

cdef class Oscillator:
    cdef cinter.Oscillator* _osc
    cdef wrap(self,cinter.Oscillator * osc)

cdef class OscillatorNetwork:
    cdef cinter.OscillatorNetwork * _oscillator_network
    cdef cinter.EvolutionContext * _evo
    cdef public list oscillators

    # This is for bug #37 in homogeneous networks
    cdef ParaMap _paramap

