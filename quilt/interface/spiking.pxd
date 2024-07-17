# distutils: language = c++
cimport cinterface as cinter
cimport base


cdef class SparseProjector:

    cdef:
        cinter.SparseProjection * _projection
        int type
        int dist_type
        double connectivity
        float weight, weight_delta, delay, delay_delta

cdef class Population:

    cdef cinter.Population * _population
    cdef cinter.PopulationSpikeMonitor * _spike_monitor 
    cdef cinter.PopulationStateMonitor * _state_monitor

    cdef SpikingNetwork spikenet


cdef class SpikingNetwork:

    cdef cinter.SpikingNetwork * _spiking_network
    cdef cinter.EvolutionContext * _evo
    cdef str name