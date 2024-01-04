#pragma once
#include <iostream>
#include <variant>
#include <vector>

typedef  std::vector<double> neuron_state;
class Population;

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


/**
 * The input for neurons.
// */
// template <typename var>
// class Injector {
// public:
//     Injector( var * _variable_ptr, double rate, double t_max)
//         : variable_ptr(_variable_ptr), rate(rate), t_max(t_max) {}

//     virtual void inject(EvolutionContext *evo) {
//         // if (evo->now < t_max && variable_ptr) {
//         //     *variable_ptr += rate * evo->dt;
//         // }
//     }

//     protected:
//         var * variable_ptr;
//         double rate, t_max;
// };


class PopCurrentInjector{
    public:
        PopCurrentInjector(Population * pop, double I, double t_max):pop(pop), I(I), t_max(t_max){}
        void inject(EvolutionContext * evo);
        
    private:
        Population * pop;
        double I, t_max;
};