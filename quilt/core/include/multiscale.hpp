#include "base.hpp"
#include "network.hpp"
#include "oscillators.hpp"
#include "devices.hpp"

#include <vector>

using std::vector;

class Transducer{
    public:
        vector<dynamical_state> state_history;

        PoissonSpikeSource * injector;
        PopulationSpikeMonitor * monitor;

        Transducer(Population * population, const ParaMap * params);

        void evolve();
        double get_past(unsigned int axis, double time);
        
        void set_evolution_context(EvolutionContext * evo)
        {
            this->evo = evo;
        }
    private:
        Population * population;  
        EvolutionContext * evo;
};

class MultiscaleNetwork{
    public:
        SpikingNetwork * spikenet;
        OscillatorNetwork * oscnet;
        vector<Transducer> transducers;

        MultiscaleNetwork(SpikingNetwork * spikenet, OscillatorNetwork * oscnet);
    private:
        unsigned int n_populations;
        unsigned int n_oscillators;
};