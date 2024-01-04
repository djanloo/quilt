from libcpp.vector cimport vector

cdef extern from "../src_cpp/include/base_objects.hpp":
    cdef cppclass EvolutionContext:
        double dt
        double now
        EvolutionContext(double _dt)

    cdef cppclass HierarchicalID:
        pass

cdef extern from "../src_cpp/include/devices.hpp":
    cdef cppclass PopulationSpikeMonitor:
        PopulationMonitor(Population * pop)
        void gather()
        vector[int] get_history()
    
    cdef cppclass PopulationStateMonitor:
        PopulationMonitor(Population * pop)
        void gather()
        vector[vector[vector[double]]] get_history()
    
    cdef cppclass PopCurrentInjector:
        PopCurrentInjector(Population * pop, double I, double t_max)
        void inject(EvolutionContext * evo)

cdef extern from "../src_cpp/include/neurons.hpp":
    cdef cppclass neuron_type:
        pass

cdef extern from "../src_cpp/include/neurons.hpp" namespace "neuron_type":
    cdef neuron_type dummy
    cdef neuron_type aqif
    cdef neuron_type izhikevich
    cdef neuron_type aeif

cdef extern from "../src_cpp/include/network.hpp":
    cdef cppclass Projection:
        int start_dimension, end_dimension
        Projection(double ** weights, double ** delays, int start_dimension, int end_dimension)

    cdef cppclass Population:
        int n_neurons
        HierarchicalID * id

        int n_spikes_last_step

        Population(int n_neurons, neuron_type nt, SpikingNetwork * spiking_network)
        void project(Projection * projection, Population * child_pop)
        void evolve(EvolutionContext * evo)

    cdef cppclass SpikingNetwork:
        HierarchicalID * id
        SpikingNetwork()

        # I/O
        void add_injector(PopCurrentInjector * injector)
        PopulationSpikeMonitor * add_spike_monitor(Population * population)
        PopulationStateMonitor * add_state_monitor(Population * population)

        void run(EvolutionContext * evo, double time)