/**
 * Stuff for oscillators. First I try a sparse version.
 * I'm really aware that a dense "matrix" version of this would 
 * probably be better, but let's see.
 * 
 * Numerical methods for delay differential equations Bellen Zennaro
*/
#pragma once
#include "base.hpp"
#include "links.hpp"
#include "network.hpp"

#include <typeinfo>
#include <memory>

class EvolutionContext;
class Population;
class Link;

using std::vector;
using std::cout;
using std::endl;
using std::shared_ptr;
using std::make_shared;
using std::runtime_error;

class OscillatorNetwork;
class Oscillator;

extern map<std::string, int> OSCILLATOR_CODES;
extern map<int, std::string> OSCILLATOR_NAMES;


/************************************************** OSCILLATOR BASE ***************************************************/
/**
 * @class Oscillator
 * @brief Base class of oscillators
 * 
 * The `evolve_state()` method defines the oscillator dynamics and must be overriden.
*/
class Oscillator{
    public:
        HierarchicalID id;
        OscillatorNetwork * oscnet;
        ContinuousRK memory_integrator;
        string oscillator_type = "base";
        unsigned int space_dimension = 2;

        vector<Link*> incoming_osc;
        ParaMap * params;

        Oscillator(ParaMap * params, OscillatorNetwork * oscnet);

        // The (virtual) evolution function is implemented as a lambda
        // so it's easier to pass it to the ContinuousRK
 
        std::function<void(const dynamical_state & x, dynamical_state & dxdt, double t)> evolve_state;

        // Getter methods
        string get_type(){return oscillator_type;}
        unsigned int get_space_dimension(){return space_dimension;}
        vector<dynamical_state> get_history()
        {
            return memory_integrator.state_history;
        }

        double get_past(unsigned int axis, double t)
        {
            return memory_integrator.get_past(axis, t);
        }

        // Setter methods
        void set_evolution_context(EvolutionContext * evo);

    private:
        EvolutionContext * evo;
};

/************************************* OSCILLATOR FACTORY ************************************************/
// Builder method
template <class OSC>
shared_ptr<Oscillator> oscillator_maker(ParaMap * params, OscillatorNetwork * osc){
    return make_shared<OSC>(params, osc);
}

class OscillatorFactory{
    typedef std::function<shared_ptr<Oscillator>(ParaMap *, OscillatorNetwork *)> constructor;
    public:
        bool add_constructor(string const& oscillator_type, constructor const& lker) {
            return _constructor_map.insert(std::make_pair(oscillator_type, lker)).second;
        }

        shared_ptr<Oscillator> get_oscillator(string const& oscillator_type, ParaMap * params, OscillatorNetwork * osc_net);
        OscillatorFactory();
    private:
        map<string, constructor> _constructor_map;
};

// Singleton method to return a unique instance of OscillatorFactory
OscillatorFactory& get_oscillator_factory();

/*************************************************** OSCILLATOR MODELS *************************************************/
class harmonic_oscillator : public Oscillator{
    public:
        float k;
        harmonic_oscillator(ParaMap * params, OscillatorNetwork * oscnet);
};

class test_oscillator : public Oscillator{
    public:
        float k;
        test_oscillator(ParaMap * params, OscillatorNetwork * oscnet);
};

class jansen_rit_oscillator : public Oscillator{
    public:
        float ke, ki, He, Hi, v0, C, s, rmax;
        jansen_rit_oscillator(ParaMap * params, OscillatorNetwork * oscnet);
        double sigm(double v);
};

class leon_jansen_rit_oscillator : public Oscillator{
    public:
        static float He, Hi, ke, ki;
        static float gamma_1, gamma_2, gamma_3, gamma_4, gamma_5;
        static float gamma_1T, gamma_2T, gamma_3T;
        static float e0, rho1, rho2;
        static float U, P, Q;
        leon_jansen_rit_oscillator(ParaMap * params, OscillatorNetwork * oscnet);
        static double sigm(double v);
};


/******************************************** NETWORK ***********************************************/
/**
 * @class OscillatorNetwork
 * @brief A network of oscillators
 * 
 * I
*/
class OscillatorNetwork{
    public:
        HierarchicalID id;

        // The homogeneous constructor
        OscillatorNetwork(int N, ParaMap * params);
        
        vector<shared_ptr<Oscillator>> oscillators;
        
        void build_connections(Projection * proj, ParaMap * link_params);
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