#include "base.hpp"
#include "network.hpp"
#include "oscillators.hpp"
#include "devices.hpp"

#include <vector>

using std::vector;
using std::shared_ptr;

class T2OLink: public Link{
    public:
         T2OLink(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay, ParaMap* params)
        : Link(source, target, weight, delay, params) {}

        double get(int axis, double now) override;
};

class O2TLink: public Link{
    public:
         O2TLink(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay, ParaMap* params)
        : Link(source, target, weight, delay, params) {}

        double get(int axis, double now) override;
};


class Transducer: public Oscillator{
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