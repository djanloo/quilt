/**
 * Stuff for oscillators. First I try a sparse version.
 * I'm really aware that a dense "matrix" version of this would 
 * probably be better, but let's see.
 * 
 * Numerical methods for delay differential equations Bellen Zennaro
*/
#pragma once
#include "base.hpp"
#include "network.hpp"

class EvolutionContext;
class Population;

using std::vector;
using std::cout;
using std::endl;

class OscillatorNetwork;

/**
 * @class Link
 * @brief Delay-weight link for oscillators
 * 
 * The main method is `get(axis, time)`, that returns the specified state variable of `dynamical_state` of `source` at \f$t = t_{now}-\tau_{i,j} \f$.
 * 
*/
template <class SOURCE, class DESTINATION>
class Link{
    public:
        SOURCE * source;
        DESTINATION * destination;
        float weight, delay;

        Link(SOURCE * source, DESTINATION * destination,float weight, float delay):
        source(source), destination(destination), weight(weight), delay(delay){}

        double get(int axis, double now);

        void set_evolution_context(EvolutionContext * evo)
        {
            this->evo = evo;
        };
    private:
        EvolutionContext * evo;
};


/**
 * @class Oscillator
 * @brief Base class of oscillators
 * 
 * The `evolve_state()` method defines the oscillator dynamics and must be overriden.
*/
class Oscillator{
    public:
        unsigned int space_dimension = 2;
        HierarchicalID id;
        OscillatorNetwork * oscnet;
        ContinuousRK memory_integrator;

        std::vector< Link<Oscillator, Oscillator>> incoming_osc;

        Oscillator(OscillatorNetwork * oscnet);
        void connect(Oscillator * osc, float weight, float delay);

        vector<dynamical_state> get_history()
        {
            return memory_integrator.state_history;
        }

        double get_past(unsigned int axis, double t)
        {
            return memory_integrator.get_past(axis, t);
        }

        // The (virtual) evolution function
        std::function<void(const dynamical_state & x, dynamical_state & dxdt, double t)> evolve_state;
        
        void set_evolution_context(EvolutionContext * evo)
        {
            this->evo = evo;
            memory_integrator.set_evolution_context(evo);
            for (auto & incoming_link : incoming_osc){
                incoming_link.set_evolution_context(evo);
            }
        };
    private:
        EvolutionContext * evo;
};


class harmonic_oscillator : public Oscillator{
    public:
        float k;
        harmonic_oscillator(const ParaMap * params, OscillatorNetwork * oscnet);
};

class test_oscillator : public Oscillator{
    public:
        float k;
        test_oscillator(const ParaMap * params, OscillatorNetwork * oscnet);
};

class jansen_rit_oscillator : public Oscillator{
    public:
        float a, b, A, B, v0, C, r, vmax;
        jansen_rit_oscillator(const ParaMap * params, OscillatorNetwork * oscnet);
        static double sigm(double v, float nu_max, float v0, float r);
};

/**
 * @class OscillatorNetwork
 * @brief A network of oscillators
 * 
 * I
*/
class OscillatorNetwork{
    public:
        HierarchicalID id;
        OscillatorNetwork():id(){};
        
        std::vector<Oscillator*> oscillators;
        
        void initialize(EvolutionContext * evo, vector<dynamical_state> init_conds);
        void run(EvolutionContext * evo, double time, int verbosity);

        void set_evolution_context(EvolutionContext * evo){
            this->evo = evo;
            for (auto & oscillator : oscillators){
                oscillator->set_evolution_context(evo);
            }
        };

    private:
        bool is_initialized = false;
        EvolutionContext * evo;
};