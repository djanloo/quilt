from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "../core/include/base.hpp":
    cdef cppclass EvolutionContext:
        double dt
        double now
        EvolutionContext(double _dt)

    cdef cppclass HierarchicalID:
        pass

    cdef cppclass ParaMap:
        ParaMap()
        void update(ParaMap * new_values) except +
        void add(string key, float value) except +
        float get(string key) const
        float get(string key, float deafaul_value) const

cdef extern from "../core/include/devices.hpp":
    cdef cppclass PopulationSpikeMonitor:
        PopulationMonitor(Population * pop)
        void gather()
        vector[int] get_history()
    
    cdef cppclass PopulationStateMonitor:
        PopulationMonitor(Population * pop)
        void gather()
        vector[vector[vector[double]]] get_history()
    
    cdef cppclass PopCurrentInjector:
        PopCurrentInjector(Population * pop, double I, double t_min, double t_max)
        # void inject(EvolutionContext * evo)
    
    cdef cppclass PoissonSpikeSource:
        PoissonSpikeSource( Population * pop,
                            float rate, float weight, float weight_delta,
                            float t_min, float t_max
                            )

cdef extern from "../core/include/neurons_base.hpp":
    cdef cppclass neuron_type:
        pass

    cdef cppclass NeuroParam:
        NeuroParam(const ParaMap &, neuron_type)

cdef extern from "../core/include/neurons_base.hpp" namespace "neuron_type":
    cdef neuron_type base_neuron
    cdef neuron_type aqif
    cdef neuron_type izhikevich
    cdef neuron_type aeif

cdef extern from "../core/include/network.hpp":
    cdef cppclass Projection:
        int start_dimension, end_dimension
        Projection(float ** weights, float ** delays, int start_dimension, int end_dimension)

    cdef cppclass SparseProjection:
        pass

    cdef cppclass SparseLognormProjection(SparseProjection): 
        SparseLognormProjection(double connectivity, int type,
                                unsigned int start_dimension, unsigned int end_dimension,
                                float weight, float weight_delta,
                                float delay, float delay_delta)

    cdef cppclass Population:
        int n_neurons
        HierarchicalID * id

        int n_spikes_last_step

        Population(int, ParaMap * , SpikingNetwork * ) except +
        # void project(Projection * , Population * )
        void project(SparseProjection *, Population *)
        void evolve(EvolutionContext * )

        void print_info()

    cdef cppclass SpikingNetwork:
        HierarchicalID * id
        unsigned int verbosity
        SpikingNetwork()

        # I/O
        void add_injector(PopCurrentInjector * injector)
        void add_injector(PoissonSpikeSource * injector)

        PopulationSpikeMonitor * add_spike_monitor(Population * population)
        PopulationStateMonitor * add_state_monitor(Population * population)

        void run(EvolutionContext * evo, double time, int verbosity) except +


#---------------------- OSCILLATORS ------------------ #

cdef extern from "../core/include/oscillators.hpp":
    cdef cppclass oscillator_type:
        pass

cdef extern from "../core/include/oscillators.hpp" namespace "osc_type":
    cdef oscillator_type harmonic

cdef extern from "../core/include/oscillators.hpp":
    cdef cppclass Oscillator:
        vector[double] state
        vector[vector[double]] get_history()
        void connect(Oscillator *, float, float)

    cdef cppclass OscillatorNetwork:
        vector [Oscillator *] oscillators # Note: this is reported as an error in my syntax highlighter, but it's right
        OscillatorNetwork()
        void run(EvolutionContext * evo, double t, int verbosity) except +
        void initialize(EvolutionContext * evo, vector[vector[double]] init_state)

    cdef cppclass harmonic_oscillator:
        harmonic_oscillator(ParaMap * params, OscillatorNetwork * oscnet)
        vector[vector[double]] get_history()
    
    cdef cppclass jansen_rit_oscillator:
        jansen_rit_oscillator(ParaMap * params, OscillatorNetwork * oscnet)
        vector[vector[double]] get_history()
