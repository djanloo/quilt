#pragma once
#include <iostream>
#include <fstream>
#include <variant>
#include <vector>

typedef std::vector<double> neuron_state;
class Population;
class EvolutionContext;

/**
 * @class PopulationSpikeMonitor
 * @brief Monitor for the variable `Population::n_spikes_last_step`
*/
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

/**
 * @class PopulationSpikeMonitor
 * @brief Monitor for the variables `Population::neurons::state`
*/
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
 * @brief Virtual base class for population injectors
*/
class PopInjector{
    public:
        PopInjector(Population * pop):pop(pop){}
        virtual ~PopInjector() = default;
        virtual void inject(EvolutionContext * /*evo*/){std::cout <<"WARNING: using virtual PopInjector::inject()" << std::endl;}
        Population * pop;
};

/**
 * @class PopCurrentInjector
 * @brief Constant current population injector
*/
class PopCurrentInjector: public PopInjector{
    public:
        PopCurrentInjector(Population * pop, float I, float t_min, float t_max): 
            PopInjector(pop), I(I), t_min(t_min), t_max(t_max), activated(false), deactivated(false){}

        void inject(EvolutionContext * evo) override;
        
    private:
        double I, t_min, t_max;
        bool activated, deactivated;
};

/**
 * @class PoissonSpikeSource
 * @brief Source of poisson-distributed spikes
 * 
 * For now only one-to-one connection is implemented
*/
class PoissonSpikeSource: public PopInjector{
    public:
        PoissonSpikeSource( Population * pop,
                            float rate, float weight,float weight_delta,
                            double t_min, double t_max);
        /**
         * Generates spikes until a spike is generated in another time bin to prevent the spike queue from being uselessly too much long.
         * 
        */
        void inject(EvolutionContext * evo) override;
    private:
        float rate;
        float weight;
        float weight_delta;
        double t_min, t_max;

        std::vector<float> weights;
        // static std::ofstream outfile;
        std::vector<double> next_spike_times;
};      