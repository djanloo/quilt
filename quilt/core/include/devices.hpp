/**
 * @file devices.hpp
 * @author Gianluca Becuzzi  (becuzzigianluca@gmail.com)
 * @brief Devices for the quilt multiscale simulator.
 * 
 * @copyright Copyright (c) 2024 Gianluca Becuzzi
 * 
 * 
 * This file contains the declarations of:
 * - monitors:
 *      - PopulationSpikeMonitor
 *      - PopulationRateMonitor
 * - injectors:
 *      - PoissonSpikeSource
 *      - InhomPoissonSpikeSource
 */
#pragma once

#include "base.hpp"

#include <iostream>
#include <fstream>
#include <variant>
#include <vector>
#include <exception>

using std::vector;
using std::endl;

typedef std::vector<double> dynamical_state;
class Population;

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

class PopulationRateMonitor : public PopulationMonitor{
    public:
        PopulationRateMonitor(Population * population)
            :   PopulationMonitor(population){}

        void gather() override;
        vector<float> get_history();
    protected:
        vector<float> history;
};

/**
 * @class PopulationSpikeMonitor
 * @brief Monitor for the variable `Population::n_spikes_last_step`
*/
class PopulationSpikeMonitor : public PopulationMonitor{
    public:

        PopulationSpikeMonitor(Population * pop)
            :   PopulationMonitor(pop){}

        void gather();
        vector<int> get_history();
    protected:
        vector<int> history;
};

/**
 * @class PopulationSpikeMonitor
 * @brief Monitor for the variables `Population::neurons::state`
*/
class PopulationStateMonitor : public PopulationMonitor{
    public:

        PopulationStateMonitor(Population * pop)
            :   PopulationMonitor(pop){}

        void gather();
        vector<vector<dynamical_state>> get_history();
    protected:
        vector<vector<dynamical_state>> history;
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
            get_global_logger().log(WARNING, "using virtual PopInjector::inject()");
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
        
        double I, t_min, t_max;
        bool activated, deactivated;
};

/** 
 * @class MonoPhasicDBSinjector
 * @brief Test for a monophasic DBS current Injector
*/
class MonoPhasicDBSinjector: public PopCurrentInjector{
    public:
        MonoPhasicDBSinjector(Population * pop, float I, float t_min, float t_max, float pulse_width, float period_width)
            :   PopCurrentInjector(pop, I, t_min, t_max),
                pulse_width(pulse_width),
                period_width(period_width){
                    // Controllo su t_min
                    if (t_min < 0) {
                        throw std::invalid_argument(
                            "t_min was set to negative (" + std::to_string(t_min) + 
                            ") in MonoPhasicDBSinjector"
                        );
                    }

                    // Se t_max < t_min, mettilo a infinito
                    if (t_max < t_min) {
                        this->t_max = std::numeric_limits<double>::infinity();
                    } else {
                        this->t_max = t_max;
                    }

                    // Controlli di validità dei parametri stimolo
                    if (pulse_width <= 0 || period_width <= 0) {
                        throw std::invalid_argument("pulse_width and period_width must be > 0");
                    }
                    if (pulse_width > period_width) {
                        throw std::invalid_argument("pulse_width cannot exceed period_width");
                    }

                    get_global_logger().log(INFO, 
                        "MonoPhasicDBSinjector initialized: I=" + std::to_string(I) +
                        " nA, pulse_width=" + std::to_string(pulse_width) +
                        " ms, period_width=" + std::to_string(period_width) +
                        " ms, t_min=" + std::to_string(t_min) +
                        ", t_max=" + (std::isinf(this->t_max) ? "inf" : std::to_string(this->t_max))
                    );
                }

        void inject(EvolutionContext * evo) override;
        
    private:
        double pulse_width, period_width;
};


/** 
 * @class BiphasicDBSinjector
 * @brief DBS current injector with biphasic, charge-balanced pulses
*/
class BiphasicDBSinjector: public PopCurrentInjector {
public:
    BiphasicDBSinjector(Population * pop, 
                        float I_pos, float I_neg, 
                        float t_min, float t_max, 
                        float pulse_width_pos, float pulse_width_neg, 
                        float period_width)
        : PopCurrentInjector(pop, I_pos, t_min, t_max),
          I_neg(I_neg),
          pulse_width_pos(pulse_width_pos),
          pulse_width_neg(pulse_width_neg),
          period_width(period_width)
    {
        // Controllo t_min
        if (t_min < 0) {
            throw std::invalid_argument(
                "t_min was set to negative (" + std::to_string(t_min) + 
                ") in BiphasicDBSinjector"
            );
        }

        // t_max fallback a infinito
        if (t_max < t_min) {
            this->t_max = std::numeric_limits<double>::infinity();
        } else {
            this->t_max = t_max;
        }

        // Controlli parametri stimolo
        if (pulse_width_pos <= 0 || pulse_width_neg <= 0 || period_width <= 0) {
            throw std::invalid_argument("pulse_width_pos, pulse_width_neg, and period_width must be > 0");
        }
        if (pulse_width_pos + pulse_width_neg > period_width) {
            throw std::invalid_argument("Sum of biphasic pulse widths cannot exceed period_width");
        }

        get_global_logger().log(INFO, 
            "BiphasicDBSinjector initialized: I_pos=" + std::to_string(I) +
            " nA, I_neg=" + std::to_string(I_neg) +
            " nA, pulse_width_pos=" + std::to_string(pulse_width_pos) +
            " ms, pulse_width_neg=" + std::to_string(pulse_width_neg) +
            " ms, period_width=" + std::to_string(period_width) +
            " ms, t_min=" + std::to_string(t_min) +
            ", t_max=" + (std::isinf(this->t_max) ? "inf" : std::to_string(this->t_max))
        );
    }

    void inject(EvolutionContext * evo) override;

private:
    double pulse_width_pos, pulse_width_neg, period_width;
    float I_neg;  // Ampiezza fase negativa
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
         * Generates spikes until a spike is generated in another time bin to prevent the spike queue from being uselessly too long.
         * 
        */
        void inject(EvolutionContext * evo) override;

        // This method is an approximation, must be removed in future
        // by building the inhomogeneous poisson spikesource
        void set_rate(float new_rate);
    private:
        float rate;         //!< Rate of the Poisson process [Hz]
        float weight;       //!< Weight of the spikes
        float weight_delta; //!< Semidispersion of the weight of the spikes
        double t_min, t_max;

        std::vector<float> weights;
        // static std::ofstream outfile;
        std::vector<double> next_spike_times;

        RNG rng;
}; 

/**
 * @class InhomPoissonSpikeSource
 * @brief Source of poisson-distributed spikes with time-dependent rate
 * 
 * 
 * For now only one-to-one connection is implemented
 * 
 * @param pop Target spiking population of the source
 * @param rate_function The rate function r(t). Must be double(double)
 * @param weight
 * @param weight_delta
 * @param generation_window_length The length of the generation windos in ms
*/
class InhomPoissonSpikeSource: public PopInjector{
    public:
        /**
         * Initializes a Inhomogeneous Poisson Spike Source
         * 
         * @param pop The spiking population
         * @param rate_function The rate function. It must have double(double) signature, given a time returns the rate. 
        */
        InhomPoissonSpikeSource( Population * pop,
                            std::function<double(double)> rate_function, 
                            float weight, float weight_delta, 
                            double generation_window_length);
        /**
         * 
         * Generate one spike for neuron given the current rate
         * 
        */
        void _inject_partition(const vector<double> &rate_buffer, double now, double dt, int start_id, int end_id, RNGDispatcher * rng_disp);
        void inject(EvolutionContext * evo) override;

        std::shared_ptr<PerformanceManager> perf_mgr;

    private:
        std::function<double(double)> rate_function;
        float weight;       //!< Weight of the spikes
        float weight_delta; //!< Semidispersion of the weight of the spikes

        std::vector<float> weights;
        static ThreadSafeFile outfile; //DEBUG
        double generation_window_length; 
        double currently_generated_time;
        RNG rng;

        std::vector<double> integration_start;
        std::vector<double> integration_leftovers;
        std::vector<double> integration_thresholds;
};   