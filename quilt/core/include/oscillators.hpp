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

extern const std::map<std::string, int> OSCILLATOR_CODE;

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
 * Necessary to implement a brutal form of polymorphism with templates
 * Please don't judge me I'm in a hurry
*/
class LinkBase{
    public:
    float weight, delay;
        LinkBase(float weight, float delay)
            :   weight(weight),
                delay(delay)
        {
            if (weight == 0.0)
            {
                throw runtime_error("Initialized a zero-weighted link between two oscillators");
            }
        }
        ~LinkBase(){}
        
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

/**
 * @class Link
 * @brief Delay-weight link for oscillators
 * 
 * The main method is `get(axis, time)`, that returns the specified state variable of `dynamical_state` of `source` at \f$t = t_{now}-\tau_{i,j} \f$.
 * 
*/
template <class SOURCE, class TARGET>
class Link : public LinkBase{
    public:
        SOURCE * source;
        TARGET * target;

        Link(SOURCE * source, TARGET * target, float weight, float delay)
            :   LinkBase(weight, delay),
                source(source), 
                target(target)
        {
            cout << "Creating a Link "<< typeid(*source).name() << " -- " << typeid(*target).name() <<endl;
        }

        double get(int axis, double now);

    private:
        EvolutionContext * evo;
};

// Factory method
template <class A, class B>
static LinkBase * make_link(A * source, B * target, float weight, float delay)
{
    return new Link<A,B>(source, target, weight, delay);
}


class Connector{
    public:
        Connector(){}
        template <class A, class B>
        void make_link(A * source, B * target, float weight, float delay)
        {
            target->incoming_osc.push_back(new Link<A,B>(source, target, weight, delay));
        }
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

        vector<LinkBase*> incoming_osc;

        Oscillator(OscillatorNetwork * oscnet);

        vector<dynamical_state> get_history()
        {
            return memory_integrator.state_history;
        }

        double get_past(unsigned int axis, double t)
        {
            return memory_integrator.get_past(axis, t);
        }

        // The (virtual) evolution function is implemented as a lambda
        // so it's easier to pass it to the ContinuousRK
        std::function<void(const dynamical_state & x, dynamical_state & dxdt, double t)> evolve_state;
        
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

class leon_jansen_rit_oscillator : public Oscillator{
    public:
        static float He, Hi, ke, ki;
        static float gamma_1, gamma_2, gamma_3, gamma_4, gamma_5;
        static float gamma_1T, gamma_2T, gamma_3T;
        static float e0, rho1, rho2;
        static float U, P, Q;
        leon_jansen_rit_oscillator(const ParaMap * params, OscillatorNetwork * oscnet);
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
        OscillatorNetwork():id(){};

        // The homogeneous constructor
        OscillatorNetwork(int N, ParaMap * params);
        
        vector<shared_ptr<Oscillator>> oscillators;
        
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