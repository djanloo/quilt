#pragma once
#include <iostream>
#include <variant>
#include <vector>

typedef std::vector<double> neuron_state;
class Population;
class EvolutionContext;

class PopulationSpikeMonitor{
    public:

        PopulationSpikeMonitor(Population * pop){this->monitored_pop = pop;};
        void gather();

        std::vector<int> get_history(){
            return this->history;
        }
    private:
        Population * monitored_pop;
        std::vector<int> history;
};

class PopulationStateMonitor{
    public:

        PopulationStateMonitor(Population * pop){this->monitored_pop = pop;};
        void gather();

        std::vector<std::vector<neuron_state>> get_history(){
            return this->history;
        }
    private:
        Population * monitored_pop;
        std::vector<std::vector<neuron_state>> history;
};


class PopCurrentInjector{
    public:
        PopCurrentInjector(Population * pop, float I, float t_min, float t_max):
        pop(pop), I(I), t_min(t_min), t_max(t_max), activated(false), deactivated(false){}
        void inject(EvolutionContext * evo);
        
    private:
        Population * pop;
        double I, t_min, t_max;
        bool activated, deactivated;
};