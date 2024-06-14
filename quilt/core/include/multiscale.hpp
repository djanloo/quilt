#include "base.hpp"
#include "network.hpp"
#include "oscillators.hpp"
#include "devices.hpp"

#include <vector>

using std::vector;
using std::shared_ptr;

class T2JRLink: public Link{
    public:
         T2JRLink(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay, ParaMap* params)
        : Link(source, target, weight, delay, params) {}

        double get(int axis, double now) override;
};

class JR2TLink: public Link{
    public:
         JR2TLink(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay, ParaMap* params)
        : Link(source, target, weight, delay, params) {}

        double get(int axis, double now) override;
};


class Transducer: public Oscillator{
    public:
        PoissonSpikeSource * injector;
        PopulationSpikeMonitor * monitor;

        MultiscaleNetwork * multinet;

        /**
         * @brief Transducers are the bridge between spiking and neural mass.
         * 
         * Is composed of a PoissonSpikeSource and a PopulationSpikeMonitor
         * 
        */
        Transducer(Population * population, ParaMap * params, MultiscaleNetwork * multinet);

        void evolve();

        /**
         * The get_past method is used by Oscillators to get the input in the DDEs.
         * It must thus return the rate of the spiking population in the past
        */
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
        unsigned int time_ratio;

        MultiscaleNetwork(SpikingNetwork * spikenet, OscillatorNetwork * oscnet, unsigned int time_ratio);
        void run(EvolutionContext * evo, double time, int verbosity);

    private:
        unsigned int n_populations;
        unsigned int n_oscillators;
};