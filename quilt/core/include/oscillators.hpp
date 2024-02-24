/**
 * Stuff for oscillators. First I try a sparse version.
 * I'm really aware that a dense "matrix" version of this would 
 * probably be better, but let's see.
 * 
 * Numerical methods for delay differential equations Bellen Zennaro
*/
#pragma once
#include "base_objects.hpp"
#include "network.hpp"

class EvolutionContext;
class Population;

using std::vector;
using std::cout;
using std::endl;

class OscillatorNetwork;

/**Available type of oscillators*/
enum class oscillator_type : unsigned int {base_oscillator, harmonic, jensen_rit, red_wong_wang};


/**
 * @class Link
 * @brief Delay-weight link for oscillators
 * 
 * The main method is `get()`, that returns the `dynamical_state` of `source` at \f$t = t_{now}-\tau_{i,j} \f$.
 * 
 * This is a patchwork for now. DDEs are not a just ODEs with a-posteriori interpolation.
*/
template <class SOURCE, class DESTINATION>
class Link{
    public:
        SOURCE * source;
        DESTINATION * destination;
        float weight, delay;
        static float timestep;

        Link(SOURCE * source, DESTINATION * destination, 
                float weight, float delay,
                EvolutionContext * evo
                ):
        source(source), destination(destination),weight(weight),delay(delay),evo(evo){}
        double get(int axis, double now);
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
        ContinuousRK memory_integrator;

        unsigned int space_dimension = 2;
        static dynamical_state none_state;    //!< This is temporary! The problem starts in C[-T, 0]
        oscillator_type osc_type = oscillator_type::base_oscillator;
        HierarchicalID id;
        OscillatorNetwork * oscnet;

        std::vector< Link<Oscillator, Oscillator>> incoming_osc;

        Oscillator(OscillatorNetwork * oscnet, EvolutionContext * evo);
        void connect(Oscillator * osc, float weight, float delay);
        
        std::function<void(const dynamical_state & x, dynamical_state & dxdt, double t)> evolve_state;
    private:
        EvolutionContext * evo;
};

/**
 * @class spiking_oscillator
 * @brief An oscillator linked to a spiking population
*/
class spiking_oscillator : public Oscillator{
    public:
        Population * population;

        /**
         * 
         * @brief Coarse-grain transformer function
         * 
         * This function  must mirror the evolution of a population, i.e. 
         * it must perform, if \f$\gamma(t)\f$ is the state of the spiking population:
         * 
         * @f[
         *      \Gamma(t+dt_{osc}) = F(\gamma(t), \gamma(t - dt_{spiking}, .., \gamma(t- n \cdot dt_{spiking})) 
         * @f]
         */
        void evolve_state();
};


/**
 * @class harmonic
 * @brief test harmonic oscillator
 * 
 * Must be removed in future.
*/
class harmonic_oscillator : public Oscillator{
    public:
        float k;
        harmonic_oscillator(const ParaMap * params, OscillatorNetwork * oscnet, EvolutionContext * evo);
};

class test_oscillator : public Oscillator{
    public:
        float k;
        test_oscillator(const ParaMap * params, OscillatorNetwork * oscnet, EvolutionContext * evo);
};

class jansen_rit_oscillator : public Oscillator{
    public:
        float a, b, A, B, v0, C, r, vmax;
        jansen_rit_oscillator(const ParaMap * params, OscillatorNetwork * oscnet, EvolutionContext * evo);
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
        OscillatorNetwork(EvolutionContext * evo):id(), evo(evo){};
        
        std::vector<Oscillator*> oscillators;
        
        void init_oscillators(vector<dynamical_state> init_conds);
        void run(EvolutionContext * evo, double time);
    private:
        EvolutionContext * evo;
};