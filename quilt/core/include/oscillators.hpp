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

#include <typeinfo>
#include <memory>

class EvolutionContext;
class Population;

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

/*************************************************** LINK BASE ************************************************/

// NOTE: the linking strategy is still 'not elegant' (pronounced 'notto ereganto' with the voice of Housemaster Henry Henderson from Spy x Family)
// 
// My intent was finding a strategy to template & polymorph stuff
// in order to reduce to the minimum value the number of templates in cython code.

// Clearly, the ideal minimum is 0.
// I will consider myself satisfied only when this minimum is reached, i.e. 
// when the code will deduce the right link function when just:
// 
// osc1.connect(osc2)
// 
// is called.

/**
 * @brief Base class of links
 * 
*/
class Link{
    public:
        shared_ptr<Oscillator> source;
        shared_ptr<Oscillator> target;
        float weight, delay;
        Link(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay)
            :   source(source),
                target(target),
                weight(weight),
                delay(delay)
        {
            if (weight == 0.0)
            {
                // No link-making procedure must arrive at this point
                // zero-valued links must be treated upstream
                throw runtime_error("Initialized a zero-weighted link between two oscillators");
            }
        }
        ~Link(){}
        
        virtual double get(int axis, double now) // Note: it needs `now` for the innner steps of RK 
        {
            throw runtime_error("Using virtual `get()` of LinkBase");
        };
        
        void set_evolution_context(EvolutionContext * evo)
        {
            this->evo = evo;
        };
    protected:
        EvolutionContext * evo;
};

/*************************************** LINK FACTORY *************************************/
// Builder method for Link-derived objects
template <typename DERIVED>
Link * link_maker(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay)
{
    return new DERIVED(source, target, weight, delay);
}

class LinkFactory{
    typedef std::function<Link*(shared_ptr<Oscillator>, shared_ptr<Oscillator>, float, float)> linker;
    public:
        LinkFactory();
        bool add_linker(std::pair<string, string> const& key, linker const& lker) {
            return _linker_map.insert(std::make_pair(key, lker)).second;
        }

        Link * get_link(shared_ptr<Oscillator> source,shared_ptr<Oscillator> target, float weight, float delay);

    private:
        map<std::pair<string, string>, linker> _linker_map;
};

// Singleton method to return a unique instance of LinkFactory
LinkFactory& get_link_factory();

/****************************************************** LINK MODELS ****************************************************/
class JRJRLink : public Link{
    public:
        JRJRLink(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay)
            :   Link(source, target, weight, delay){cout << "Making JRJR link" << endl;}
        double get(int axis, double now) override;
};

class LJRLJRLink : public Link{
    public:
        LJRLJRLink(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay)
            :   Link(source, target, weight, delay){cout << "Making LJRLJR link" << endl;}
        double get(int axis, double now) override;
};

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

        vector<Link*> incoming_osc;
        const ParaMap * params;

        Oscillator(const ParaMap * params, OscillatorNetwork * oscnet);

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
        void set_evolution_context(EvolutionContext * evo)
        {
            this->evo = evo;
            memory_integrator.set_evolution_context(evo);
            for (auto & incoming_link : incoming_osc)
            {
                incoming_link->set_evolution_context(evo);
            }
        };

    private:
        const unsigned int space_dimension = 2;
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
        harmonic_oscillator(const ParaMap * params, OscillatorNetwork * oscnet);
    private:
        
        const unsigned int space_dimension = 2;
};

class test_oscillator : public Oscillator{
    public:
        float k;
        test_oscillator(const ParaMap * params, OscillatorNetwork * oscnet);
    private:
        const unsigned int space_dimension = 6;
};

class jansen_rit_oscillator : public Oscillator{
    public:
        float a, b, A, B, v0, C, r, vmax;
        jansen_rit_oscillator(const ParaMap * params, OscillatorNetwork * oscnet);
        static double sigm(double v, float nu_max, float v0, float r);
    private:
        const unsigned int space_dimension = 6;
};

class leon_jansen_rit_oscillator : public Oscillator{
    public:
        static float He, Hi, ke, ki;
        static float gamma_1, gamma_2, gamma_3, gamma_4, gamma_5;
        static float gamma_1T, gamma_2T, gamma_3T;
        static float e0, rho1, rho2;
        static float U, P, Q;
        leon_jansen_rit_oscillator(const ParaMap * params, OscillatorNetwork * oscnet);
        static double sigm(double v);
    private:
        const unsigned int space_dimension = 12;
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
        // OscillatorNetwork():id(){};

        // The homogeneous constructor
        OscillatorNetwork(int N, ParaMap * params);
        
        vector<shared_ptr<Oscillator>> oscillators;
        
        void build_connections(Projection * proj);
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