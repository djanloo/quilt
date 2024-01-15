# distutils: language = c++
"""Interface module between C++ and python.

Convention:
    - Python-visible objects get nice names
    - C++ objects get a '_' prefix
"""

from time import time
from libc.stdlib cimport malloc
from libcpp.vector cimport vector

import ctypes

import numpy as np
cimport numpy as np

cimport quilt.interface.cinterface as cinter
cimport quilt.interface.base_objects as base_objects

ctypedef vector[double] neuron_state

cdef class Projection:

    cdef int start_dimension, end_dimension
    cdef float ** _weights
    cdef float ** _delays 
    cdef cinter.Projection * _projection

    cdef float [:,:] weights, delays

    def __cinit__(self,  np.ndarray[np.float32_t, ndim=2,mode='c'] weights, np.ndarray[np.float32_t,ndim=2,mode='c'] delays):
        self.weights = weights
        self.delays = delays

        self.start_dimension = weights.shape[0]
        self.end_dimension   = weights.shape[1]

        cdef np.ndarray[np.float32_t, ndim=2, mode="c"] contiguous_weights = np.ascontiguousarray(weights, dtype = ctypes.c_float)
        cdef np.ndarray[np.float32_t, ndim=2, mode="c"] contiguous_delays  = np.ascontiguousarray(delays,  dtype = ctypes.c_float)

        self._weights = <float **> malloc(self.start_dimension * sizeof(float*))
        self._delays  = <float **> malloc(self.start_dimension * sizeof(float*))

        if not self._weights or not self._delays:
            raise MemoryError

        cdef int i
        for i in range(self.start_dimension):
            self._weights[i] = &contiguous_weights[i, 0]
            self._delays[i] = &contiguous_delays[i,0]

        self._projection = new cinter.Projection(  <float**> &self._weights[0], 
                                            <float**> &self._delays[0], 
                                            self.start_dimension, 
                                            self.end_dimension)
    @property
    def weights(self):
        return self.weights
    
    @property
    def delays(self):
        return self.delays


cdef class Population:

    cdef cinter.Population * _population
    cdef cinter.PopulationSpikeMonitor * _spike_monitor 
    cdef cinter.PopulationStateMonitor * _state_monitor

    cdef SpikingNetwork spikenet

    def __cinit__(self, int n_neurons, base_objects.ParaMap params, SpikingNetwork spikenet):

        self.spikenet = spikenet
        self._population = new cinter.Population(<int>n_neurons, params._paramap, spikenet._spiking_network)
        self._spike_monitor = NULL
        self._state_monitor = NULL

    @property
    def n_neurons(self):
        return self._population.n_neurons

    @property
    def n_spikes_last_step(self):
        return self._population.n_spikes_last_step

    def project(self, Projection proj, Population efferent_pop):
        self._population.project(proj._projection, efferent_pop._population)

    def add_injector(self, I, t_min, t_max):
        cdef cinter.PopCurrentInjector * injector = new cinter.PopCurrentInjector(self._population, I, t_min, t_max)
        self.spikenet._spiking_network.add_injector(injector)

    def monitorize_spikes(self):
        self._spike_monitor = self.spikenet._spiking_network.add_spike_monitor(self._population)
    
    def monitorize_states(self):
        self._state_monitor = self.spikenet._spiking_network.add_state_monitor(self._population)
    
    def get_data(self):
        data = dict()

        if self._spike_monitor:
            data['spikes'] = self._spike_monitor.get_history()

        if self._state_monitor:
            data['states'] = self._state_monitor.get_history()
        
        return data

    def __dealloc__(self):
        # print("Deallocating a population")
        if self._population != NULL:
            del self._population

cdef class SpikingNetwork:

    cdef cinter.SpikingNetwork * _spiking_network
    cdef cinter.EvolutionContext * _evo
    cdef str name

    def __cinit__(self, str name):
        self._spiking_network = new cinter.SpikingNetwork()
        self.name = name


    def run(self, dt=0.1, time=1):
        self._evo = new cinter.EvolutionContext(dt)
        self._spiking_network.run(self._evo, time)

    def __dealloc__(self):
        del self._spiking_network
        del self._evo


class RandomProjector:
    """Not an interface nor a C++ object, but stick to the convention"""
    def __init__(self,  inh_fraction=0.0, exc_fraction=0.0, 
                        max_inh = 0.1, max_exc=0.1, 
                        min_delay=0.3, max_delay=0.5):

        assert max_inh > 0, "Inhibition weight is a positive number"
        self.max_inh = max_inh
        self.max_exc = max_exc
        self.exc_fraction = exc_fraction
        self.inh_fraction = inh_fraction
        self.min_delay = min_delay
        self.max_delay = max_delay

    def get_projection(self, Population pop1, Population pop2):

        N, M = pop1.n_neurons, pop2.n_neurons

        active_inh_syn = (np.random.uniform(0,1, size=(N,M)) < self.inh_fraction)
        active_exc_syn = (np.random.uniform(0,1, size=(N,M)) < self.exc_fraction)

        weights = np.zeros((N,M))
        delays = np.zeros((N,M))

        start = time()

        exc_weights = np.random.uniform(0, self.max_exc, size=(N,M))
        inh_weights = np.random.uniform(0, self.max_inh, size=(N,M))

        exc_weights[~active_exc_syn] = 0.0
        inh_weights[~active_inh_syn] = 0.0
        weights = exc_weights - inh_weights

        delays = np.random.uniform(self.min_delay, self.max_delay, size=(N,M))
        delays[(~active_inh_syn)&(~active_exc_syn)] = 0.0

        end = time()
        print(f"Generating weights and delays took {end-start:.3f} seconds")
        print(f"Min delay {np.min(delays[delays != 0])}")

        self.last_weights = weights
        self.last_delays = delays

        self.last_projection = Projection(weights.astype(np.float32), delays.astype(np.float32))
        return self.last_projection
