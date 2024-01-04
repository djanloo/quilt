#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <variant>

#define WEIGHT_EPS 0.00001

// The menu

class HierarchicalID;
class EvolutionContext;

class Neuron;

class Projection;
class Population;
class SpikingNetwork;

class PopulationSpikeMonitor;
class PopulationStateMonitor;
class PopCurrentInjector;


class Projection{
    public:
        int start_dimension, end_dimension;
        double ** weights, **delays;

        Projection(double ** weights, double ** delays, int start_dimension, int end_dimension);
};

class Population{
    public:
        int n_neurons;
        std::vector<Neuron*> neurons;
        HierarchicalID * id;

        // Biophysical attributes
        int n_spikes_last_step;

        Population(int n_neurons, neuron_type nt, SpikingNetwork * spiking_network);
        void project(Projection * projection, Population * child_pop);
        void evolve(EvolutionContext * evo);
        int monitor();
};

class SpikingNetwork{
    public:
        std::vector<Population*> populations;
        HierarchicalID * id;
        SpikingNetwork();

        // Injectors (inputs)
        std::vector<PopCurrentInjector*> injectors;
        void add_injector(PopCurrentInjector * injector){this->injectors.push_back(injector);}

        // Monitors (outputs)
        std::vector<PopulationSpikeMonitor*> population_spike_monitors;
        std::vector<PopulationStateMonitor*> population_state_monitors;

        PopulationSpikeMonitor * add_spike_monitor(Population * population);
        PopulationStateMonitor * add_state_monitor(Population * population);

        // Evolution stuff
        void evolve(EvolutionContext * evo);
        void run(EvolutionContext * evo, double time);
};