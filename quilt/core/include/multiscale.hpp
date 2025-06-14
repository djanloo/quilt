#pragma once
#include "links.hpp"
#include "oscillators.hpp"

#include <memory>
#include <vector>

using std::vector;
using std::shared_ptr;

// Forward declarations
class Population;
class InhomPoissonSpikeSource;
class PopulationSpikeMonitor;
class SpikingNetwork;
class MultiscaleNetwork;

/**
 * @class Transducer
 * @brief Bridge object between spiking populations and neural mass oscillators
 * 
 * It is composed by a PoissonSpikeSource (in future PoissonInhomogeneousSpikeSource)
 * and a SpikeMonitor.
 * 
 * Implements averaging on the slow timescale (`get_past()` method).
 * 
*/
class Transducer: public Oscillator{
    private:
        Population * population;  
        double initialization_rate;
        // static ThreadSafeFile outfile;

    public:
        InhomPoissonSpikeSource * injector;
        PopulationSpikeMonitor * monitor;

        MultiscaleNetwork * multinet;

        /**
         * @brief Transducers are the bridge between spiking and neural mass.
         * 
         * Is composed of a PoissonSpikeSource and a PopulationSpikeMonitor
         * 
        */
        Transducer(Population * population, ParaMap * params, MultiscaleNetwork * multinet);
        ~Transducer();


        /**
         * O->T
         * 
         * the incoming_rate() function collects the rate of all the incoming links
         * from neural mass oscillators. It's used as a rate function of the InhomogeneousPoissonSpikeSource,
         * thus must be a double(double) function.
         * 
        */
        double neural_mass_rate(double now);

        /**
         * T -> O
         * 
         * The get_past() method is used by Oscillators to get the input in the DDEs due to the tranducer activity.
         * It must thus return the rate of the spiking population in the past.
        */
        double spiking_pop_rate(double time);

        /** 
         * Explicitly overrides non-usable oscillator methods.
         */
        double get_past(unsigned int /*axis*/, double /*time*/) override {
            throw runtime_error("Transducer::get_past(double) cannot be called.\n Use Transducer::spiking_pop_rate(time) or Transducer::neural_mass_rate(now).\nIf you see this error, SOMETHING IS REALLY MESSED UP.");
        }

        vector<double> history;

};

class MultiscaleNetwork{
    public:
        SpikingNetwork * spikenet;
        OscillatorNetwork * oscnet;
        vector<Transducer *> transducers;

        MultiscaleNetwork(SpikingNetwork * spikenet, OscillatorNetwork * oscnet);

        void build_multiscale_projections(Projection * projS2O, Projection * projO2S);

        void run(double time, int verbosity);
        void set_evolution_contextes(EvolutionContext * evo_short, EvolutionContext * evo_long);

        void add_transducer(Population * population, ParaMap * params);

        unsigned int n_populations; //!< Number of populations in the multiscale network. Must not change after init.
        unsigned int n_oscillators; //!< Number of oscillators in the multiscale network. Must not change after init.
        unsigned int time_ratio;    //!< Timescale separation. Must be defined by two evolution contextes.

        std::shared_ptr<PerformanceManager> perf_mgr;
    private:
        bool timescales_initialized;
        bool multiscale_connections_done;
        EvolutionContext * evo_short;
        EvolutionContext * evo_long;
};

/********************************* LINKS *******************************/
/**
 * @brief Link between a transducer and a Jansen-Rit oscillator
 * 
*/
class T2JRLink: public Link{
    public:
         T2JRLink(Oscillator * source, Oscillator * target, float weight, float delay, ParaMap* params)
        : Link(source, target, weight, delay, params) {}

        /**
         * Returns the firing rate of the population linked to the transducer.
         * Firign rate is averaged on the long timescale.
         * 
         * Note: the averaging is performed by `Transducer::get_past()` method
        */
        double get_rate(double now) override;
};

/**
 * @brief Link between a Jansen-Rit oscillator and a transducer
*/
class JR2TLink: public Link{
    public:
         JR2TLink(Oscillator * source, Oscillator * target, float weight, float delay, ParaMap* params)
        : Link(source, target, weight, delay, params) {}

        double get_rate(double now) override;
};

/**
 * @brief Link between a transducer and a Noisy-Jansen-Rit oscillator
 * 
*/
class T2NJRLink: public Link{
    public:
         T2NJRLink(Oscillator * source, Oscillator * target, float weight, float delay, ParaMap* params)
        : Link(source, target, weight, delay, params) {}

        /**
         * Returns the firing rate of the population linked to the transducer.
         * Firign rate is averaged on the long timescale.
         * 
         * Note: the averaging is performed by `Transducer::get_past()` method
        */
        double get_rate(double now) override;
};

/**
 * @brief Link between a Noisy-Jansen-Rit oscillator and a transducer
*/
class NJR2TLink: public Link{
    public:
         NJR2TLink(Oscillator * source, Oscillator * target, float weight, float delay, ParaMap* params)
        : Link(source, target, weight, delay, params) {}

        double get_rate(double now) override;
};


/**
 * @brief Link between a transducer and a BiNoisy-Jansen-Rit oscillator
 * 
*/
class T2BNJRLink: public Link{
    public:
         T2BNJRLink(Oscillator * source, Oscillator * target, float weight, float delay, ParaMap* params)
        : Link(source, target, weight, delay, params) {}

        /**
         * Returns the firing rate of the population linked to the transducer.
         * Firign rate is averaged on the long timescale.
         * 
         * Note: the averaging is performed by `Transducer::get_past()` method
        */
        double get_rate(double now) override;
};

/**
 * @brief Link between a BiNoisy-Jansen-Rit oscillator and a transducer
*/
class BNJR2TLink: public Link{
    public:
        BNJR2TLink(Oscillator * source, Oscillator * target, float weight, float delay, ParaMap* params)
        : Link(source, target, weight, delay, params) {}

        double get_rate( double now) override;
};
