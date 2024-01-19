#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <variant>
#include "base_objects.hpp"

#define WEIGHT_EPS 0.00001

// The menu
class HierarchicalID;
class EvolutionContext;

class Neuron;
class NeuroParam;
enum class neuron_type : unsigned int;

class Projection;
class Population;
class SpikingNetwork;

class PopulationSpikeMonitor;
class PopulationStateMonitor;
class PopInjector;

/**
 * @class Projection
 * @brief Implements a projection between to populations
 * 
 * @param[in] weights, delays
 * @param[in] start_dimension, end_dimension
 * 
 * This class is not really useful anymore and will be deprecated.
 * 
*/
class Projection{
    public:
        float ** weights, **delays;
        unsigned int start_dimension, end_dimension;

        Projection(float ** weights, float ** delays, unsigned int start_dimension, unsigned int end_dimension);
};


/**
 * @class Population
 * @brief A collection of neurons sharing neuron model
 * 
 * @param[in] n_neurons, neur_type, spiking_network
 * 
 * This class server the purpose of storing quantities that are
 * related to a group of neurons rather than a neuron itself:
 * 
 * @param n_spikes_last_step 
 * 
*/
class Population{
    public:
        int n_neurons;
        std::vector<Neuron*> neurons;
        HierarchicalID id;
        double timestats_evo;
        double timestats_spike_emission;

        // Biophysical attributes
        int n_spikes_last_step;
        NeuroParam * neuroparam;
        
        Population(int n_neurons, ParaMap * params, SpikingNetwork * spiking_network);
        void project(Projection * projection, Population * child_pop);
        void evolve_bunch(EvolutionContext * evo, unsigned int from, unsigned int to);
        void evolve(EvolutionContext * evo);
        void print_info();
};

/**
 * @class SpikingNetwork
 * @brief A collection of populations
 * 
 * This class is intended to manage input/output to/from populations
 * using the objects contained in devices.cpp
 * 
 * Also, coordinates the evolution of the populations.
*/

class SpikingNetwork{
    public:
        std::vector<Population*> populations;
        HierarchicalID * id;
        SpikingNetwork();

        // Injectors (inputs)
        std::vector<PopInjector*> injectors;
        void add_injector(PopInjector * injector){this->injectors.push_back(injector);}

        // Monitors (outputs)
        std::vector<PopulationSpikeMonitor*> population_spike_monitors;
        std::vector<PopulationStateMonitor*> population_state_monitors;

        PopulationSpikeMonitor * add_spike_monitor(Population * population);
        PopulationStateMonitor * add_state_monitor(Population * population);

        // Evolution stuff
        void run(EvolutionContext * evo, double time);
};