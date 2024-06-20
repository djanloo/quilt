#pragma once
#include "base.hpp"
#include "oscillators.hpp"

#include <memory>
#include <vector>

using std::vector;
using std::shared_ptr;

// Forward declarations
class Population;
class PoissonSpikeSource;
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
        EvolutionContext * evo;

        double initialization_rate;

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
        double incoming_rate(double now);

        /**
         * T -> O
         * 
         * The get_past() method is used by Oscillators to get the input in the DDEs due to the tranducer activity.
         * It must thus return the rate of the spiking population in the past.
        */
        double get_past(unsigned int axis, double time);
        
        void set_evolution_context(EvolutionContext * evo)
        {
            this->evo = evo;
        }
    
};

class MultiscaleNetwork{
    public:
        SpikingNetwork * spikenet;
        OscillatorNetwork * oscnet;
        vector<shared_ptr<Transducer>> transducers;

        MultiscaleNetwork(SpikingNetwork * spikenet, OscillatorNetwork * oscnet);
        void build_OT_projections(Projection * projT2O, Projection * projO2T);
        void run(double time, int verbosity);
        void set_evolution_contextes(EvolutionContext * evo_short, EvolutionContext * evo_long);

        unsigned int n_populations; //!< Number of populations in the multiscale network. Must not change after init.
        unsigned int n_oscillators; //!< Number of oscillators in the multiscale network. Must not change after init.
        unsigned int time_ratio;    //!< Timescale separation. Must be defined by two evolution contextes.
    private:
        bool timescales_initialized;
        EvolutionContext * evo_short;
        EvolutionContext * evo_long;
};