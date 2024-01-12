from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "../src_cpp/include/base_objects.hpp":
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
        PopCurrentInjector(Population * pop, double I, double t_min, double t_max)
        void inject(EvolutionContext * evo)

cdef extern from "../src_cpp/include/neurons_base.hpp":
    cdef cppclass neuron_type:
        pass

    cdef cppclass NeuroParam:
        NeuroParam(const ParaMap &, neuron_type)

cdef extern from "../src_cpp/include/neuron_models.hpp":

    cdef cppclass aeif_param(NeuroParam):
        aeif_param (const ParaMap &)


cdef extern from "../src_cpp/include/neurons_base.hpp" namespace "neuron_type":
    cdef neuron_type base_neuron
    cdef neuron_type aqif
    cdef neuron_type izhikevich
    cdef neuron_type aeif

cdef extern from "../src_cpp/include/network.hpp":
    cdef cppclass Projection:
        int start_dimension, end_dimension
        Projection(float ** weights, float ** delays, int start_dimension, int end_dimension)

    cdef cppclass Population:
        int n_neurons
        HierarchicalID * id

        int n_spikes_last_step

        Population(int, ParaMap * , SpikingNetwork * ) except +
        void project(Projection * , Population * )
        void evolve(EvolutionContext * )

    cdef cppclass SpikingNetwork:
        HierarchicalID * id
        SpikingNetwork()

        # I/O
        void add_injector(PopCurrentInjector * injector)
        PopulationSpikeMonitor * add_spike_monitor(Population * population)
        PopulationStateMonitor * add_state_monitor(Population * population)

        void run(EvolutionContext * evo, double time) except +

cdef extern from "../src_cpp/include/oscillators.hpp":
    cdef cppclass Oscillator:
        vector[double] state
        vector[vector[double]] history

        void connect(Oscillator * osc, float weight, float delay)
        void evolve(EvolutionContext * evo)

    cdef cppclass OscillatorNetwork:
        vector[Oscillator] oscillators
        OscillatorNetwork()
        void add_oscillator(Oscillator * oscillator)
        void run(EvolutionContext * evo, double t)

    cdef cppclass dummy_osc:
        vector[double] state
        vector[vector[double]] history

        dummy_osc(float k, double x, double v)
        void connect(Oscillator * osc, float weight, float delay)
        void evolve(EvolutionContext * evo)
        