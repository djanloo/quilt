from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map

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

        # The ridiculous trio
        void add_string(string key, string value) except +
        void add_float(string key, float value) except +
        # void add_int(string key, int value) except +

        # Get: raises an error if key not found
        float get(string key) except +

        # Defaulted get: sets the value in the map if not found
        float get(string key, float default_value)

cdef extern from "../core/include/devices.hpp":
    cdef cppclass PopulationSpikeMonitor:
        PopulationMonitor(Population * pop)
        vector[int] get_history()
    
    cdef cppclass PopulationStateMonitor:
        PopulationMonitor(Population *)
        vector[vector[vector[double]]] get_history()
    
    cdef cppclass PopulationRateMonitor:
        PopulationRateMonitor(Population *)
        vector[float] get_history()

    cdef cppclass PopCurrentInjector:
        PopCurrentInjector(Population * pop, double I, double t_min, double t_max)
        # void inject(EvolutionContext * evo)
    
    cdef cppclass PoissonSpikeSource:
        PoissonSpikeSource( Population * pop,
                            float rate, float weight, float weight_delta,
                            float t_min, float t_max
                            )

cdef extern from "../core/include/neurons_base.hpp":

    cdef cppclass NeuroParam:
        NeuroParam(const ParaMap &)

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
    cdef cppclass Oscillator:
        vector[vector[double]] get_history()

    cdef cppclass OscillatorNetwork:
        vector [Oscillator *] oscillators # Note: this is reported as an error in my syntax highlighter, but it's right
        
        # Homogeneous constructor: only one type of oscillator
        OscillatorNetwork(int, ParaMap *)
        # Homogeneous connections: only one type of link
        void build_connections(Projection *, ParaMap *)

        # Evolutions methods
        void run(EvolutionContext * evo, double t, int verbosity) except +
        void initialize(EvolutionContext * evo, vector[vector[double]] init_state)
