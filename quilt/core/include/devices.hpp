#pragma once
#include <iostream>
#include <fstream>
#include <variant>
#include <vector>
#include <exception>

using std::vector;
using std::cout;
using std::endl;

typedef std::vector<double> dynamical_state;
class Population;
class EvolutionContext;

/**
 * @brief Base class for population monitors
*/
class PopulationMonitor{
    public:
        PopulationMonitor(Population * population)
            :   monitored_population(population){}

        // The gather function that defines the type of monitor
        virtual void gather()
        {
            throw std::runtime_error("Used virtual `gather()` method of PopulationMonitor");
        }

        void set_evolution_context(EvolutionContext * evo)
        {
            this->evo = evo;
        }

        virtual ~PopulationMonitor(){};

    protected:
        Population * monitored_population;
        EvolutionContext * evo;
};

template <typename T>
class HistoryPopulationMonitor: public PopulationMonitor {
    public:
        HistoryPopulationMonitor(Population * pop)
            :   PopulationMonitor(pop){}

        vector<T> get_history()
        {
            return history;
        }
    protected:
        vector<T> history;
};

class PopulationRateMonitor : public HistoryPopulationMonitor<float>{
    public:
        PopulationRateMonitor(Population * population)
            :   HistoryPopulationMonitor(population){}

        void gather();
};

/**
 * @class PopulationSpikeMonitor
 * @brief Monitor for the variable `Population::n_spikes_last_step`
*/
class PopulationSpikeMonitor : public HistoryPopulationMonitor<int>{
    public:

        PopulationSpikeMonitor(Population * pop)
            :   HistoryPopulationMonitor(pop){}

        void gather();
};

/**
 * @class PopulationSpikeMonitor
 * @brief Monitor for the variables `Population::neurons::state`
*/
class PopulationStateMonitor : public HistoryPopulationMonitor<vector<dynamical_state>>{
    public:

        PopulationStateMonitor(Population * pop)
            :   HistoryPopulationMonitor(pop){}

        void gather();
};

/**
 * @brief Virtual base class for population injectors
*/
class PopInjector{
    public:
        PopInjector(Population * pop)
            :   pop(pop){}

        virtual ~PopInjector() = default;
        virtual void inject(EvolutionContext * /*evo*/)
        {
            std::cout <<"WARNING: using virtual PopInjector::inject()" << std::endl;
        }
        Population * pop;
};

/**
 * @class PopCurrentInjector
 * @brief Constant current population injector
*/
class PopCurrentInjector: public PopInjector{
    public:
        PopCurrentInjector(Population * pop, float I, float t_min, float t_max)
            :   PopInjector(pop), 
                I(I), t_min(t_min), 
                t_max(t_max), 
                activated(false), 
                deactivated(false){}

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
                            float rate, float weight, float weight_delta,
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