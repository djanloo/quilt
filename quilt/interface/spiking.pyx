# distutils: language = c++
"""Interface module between C++ and python.

Convention:
    - Python-visible objects get nice names
    - C++ objects get a '_' prefix
"""

from time import time
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

import ctypes

import numpy as np
cimport numpy as np

cimport quilt.interface.cinterface as cinter
cimport quilt.interface.base_objects as base_objects

VERBOSITY = 1

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
    
    def __dealloc__(self):
        if self._projection != NULL:
            del self._projection
        if self._weights != NULL:
            free(self._weights)
        if self._delays != NULL:
            free(self._delays)

cdef class SparseProjector():

    cdef:
        cinter.SparseProjection * _projection
        int type
        int dist_type
        double connectivity
        float weight, weight_delta, delay, delay_delta

    def __cinit__(self, params_dict, dist_type="lognorm"):
        self.type = 0 if params_dict["type"] == "exc" else 1
           
        if dist_type == "lognorm":
                self.dist_type = 0
                self.connectivity = params_dict['connectivity']
                weight, weight_delta, delay, delay_delta = map(params_dict.get, ['weight', 'weight_delta', 'delay', 'delay_delta'])
                
                if weight is None or delay is None:
                    raise KeyError("Weight and delay must be specified")
                else:
                    self.weight = weight 
                    self.delay = delay

                if weight_delta is None:
                    self.weight_delta = 0.0
                else:
                    self.weight_delta = weight_delta

                if delay_delta is None:
                    self.delay_delta = 0.0
                else:
                    self.delay_delta = delay_delta

                if self.weight < 0:
                    raise ValueError("Synaptic weight must always be positive")
                if self.delay < 0:
                    raise ValueError("Synaptic delay must always be positive")

                if self.connectivity < 0:
                    raise ValueError("Connectivity must always be > 0")
                if self.connectivity > 1:
                    raise ValueError("Connectivity must always be < 1")
            
    
    def get_projection(self, Population efferent,  Population afferent):
        self._projection = new cinter.SparseLognormProjection(self.connectivity, <int>self.type, efferent.n_neurons, afferent.n_neurons, self.weight, self.weight_delta, self.delay, self.delay_delta)
        return self

    def __dealloc__(self):
        if self._projection != NULL:
            del self._projection

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

    # def project(self, Projection proj, Population efferent_pop):
    #     self._population.project(proj._projection, efferent_pop._population)
    
    def project(self, SparseProjector proj, Population efferent_pop):
        self._population.project(proj._projection, efferent_pop._population)

    def add_const_curr_injector(self, I, t_min, t_max):
        cdef cinter.PopCurrentInjector * injector = new cinter.PopCurrentInjector(self._population, I, t_min, t_max)
        self.spikenet._spiking_network.add_injector(injector)
    
    def add_poisson_spike_injector(self, rate, weight, t_min=None, t_max=None):

        # Adapt to C++ convention that tmax< tmin means no stop
        if t_min is None:
            t_min = 0
        if t_max is None:
            t_max = -1

        if rate is None or rate < 0:
            raise ValueError(f"Invalid rate in poisson spike source: {rate}")
        
        if weight is None:
            raise ValueError(f"Invalid weight in poisson spike source: {weight}")

        cdef cinter.PoissonSpikeSource * injector = new cinter.PoissonSpikeSource(self._population, rate, weight, t_min, t_max)
        self.spikenet._spiking_network.add_injector(injector)

    def monitorize_spikes(self):
        self._spike_monitor = self.spikenet._spiking_network.add_spike_monitor(self._population)
    
    def monitorize_states(self):
        self._state_monitor = self.spikenet._spiking_network.add_state_monitor(self._population)
    
    def get_data(self, what):

        if what == "spikes":
            if self._spike_monitor == NULL:
                raise KeyError("Spikes were not monitorized for this population")
            else:
                return np.array(self._spike_monitor.get_history())
        elif what == "states":
            if self._state_monitor == NULL:
                raise KeyError("States were not monitorized for this population")
            else:
                return np.array(self._state_monitor.get_history())
        else:
            raise KeyError(f"Invalid data request to population: '{what}'")

    def print_info(self):
        self._population.print_info()

    def __dealloc__(self):
        if self._population != NULL:
            del self._population

cdef class SpikingNetwork:

    cdef cinter.SpikingNetwork * _spiking_network
    cdef cinter.EvolutionContext * _evo
    cdef str name

    def __cinit__(self, str name):
        self._spiking_network = new cinter.SpikingNetwork()
        self.name = name
        self._spiking_network.verbosity = VERBOSITY # Takes the gloabl value at creation


    def run(self, dt=0.1, time=1):
        self._spiking_network.verbosity = VERBOSITY # For last minute decisions
        self._evo = new cinter.EvolutionContext(dt)
        self._spiking_network.run(self._evo, time)

    def __dealloc__(self):
        if self._spiking_network != NULL:
            del self._spiking_network
        if self._evo != NULL:
            del self._evo

def set_verbosity(v):
    global VERBOSITY
    VERBOSITY = v

class RandomProjector:
    """Not an interface nor a C++ object, but stick to the convention"""
    def __init__(self,  inh_fraction=0.0, exc_fraction=0.0,
                        weight_inh = 0.0, weight_exc = 0.0,
                        weight_inh_delta = 0, weight_exc_delta = 0.0,
                        delay = 0.5, delay_delta=0.0):

        if weight_inh < 0:
            raise ValueError( "Inhibition weight is a positive number")

        self.weight_inh = weight_inh
        self.weight_inh_delta = weight_inh_delta

        if weight_inh - weight_inh_delta/2 < 0.0:
            raise ValueError("Inhibitory weight minimum is less than zero")

        self.weight_exc = weight_exc
        self.weight_exc_delta = weight_exc_delta

        if weight_exc - weight_exc_delta/2 < 0.0:
            raise ValueError("Excitatory weight minimum is less than zero")


        self.exc_fraction = exc_fraction
        self.inh_fraction = inh_fraction

        self.delay = delay
        self.delay_delta = delay_delta

        if weight_inh - weight_inh_delta/2 < 0.0:
            raise ValueError("Delay minimum is less than zero")

    def get_projection(self, Population pop1, Population pop2):

        N, M = pop1.n_neurons, pop2.n_neurons

        active_inh_syn = (np.random.uniform(0,1, size=(N,M)) < self.inh_fraction)
        active_exc_syn = (np.random.uniform(0,1, size=(N,M)) < self.exc_fraction)

        weights = np.zeros((N,M))
        delays = np.zeros((N,M))

        exc_weights = np.random.uniform(self.weight_exc - self.weight_exc_delta/2, self.weight_exc + self.weight_exc_delta/2, size=(N,M))
        inh_weights = np.random.uniform(0, self.weight_inh - self.weight_inh_delta/2, size=(N,M))

        exc_weights[~active_exc_syn] = 0.0
        inh_weights[~active_inh_syn] = 0.0

        weights = exc_weights - inh_weights

        delays = np.random.uniform(self.delay - self.delay_delta/2, self.delay + self.delay_delta/2, size=(N,M))
        delays[(~active_inh_syn)&(~active_exc_syn)] = 0.0


        self.last_weights = weights
        self.last_delays = delays

        self.last_projection = Projection(weights.astype(np.float32), delays.astype(np.float32))
        return self.last_projection
