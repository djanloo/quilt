/**
 * Stuff for oscillators. First I try a sparse version.
 * I'm really aware that a dense "matrix" version of this would 
 * probably be better, but let's see.
 * 
 * Numerical methods for delay differential equations Bellen Zennaro
*/
#pragma once
#include <vector>
#include <stdexcept>
#include "base_objects.hpp"
#include "network.hpp"

class EvolutionContext;
class Population;

typedef std::vector<double> osc_state;

/**Available type of oscillators*/
enum class oscillator_type : unsigned int {harmonic, jensen_rit, red_wong_wang};

/**
 * @class Link
 * @brief Delay-weight link for oscillators
 * 
 * The main method is `get()`, that retruns the `osc_state` of `source` at \f$t = t_{now}-\tau_{i,j} \f$.
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

        Link(SOURCE * source, DESTINATION * destination, float weight, float delay):
        source(source), destination(destination),weight(weight),delay(delay){}
        osc_state get(double now);
};


/**
 * @class Oscillator
 * @brief Base class of oscillators
 * 
 * The `evolve_state()` method defines the oscillator dynamics and must be overriden.
*/
class Oscillator{
    public:
        osc_state state; //!< Current state: may be moved as history.end()
        static osc_state none_state; //!< This is temporary! The problem starts in C[-T, 0]

        std::vector<osc_state> history; //!< History of the state

        std::vector< Link<Oscillator, Oscillator>> incoming_osc;

        Oscillator(){state = {0.0, 0.0};}
        void connect(Oscillator * osc, float weight, float delay);

        void evolve(EvolutionContext * evo);
        
        virtual void evolve_state(const osc_state & /*state*/, osc_state & /*dxdt*/, double /*t*/){
            throw std::runtime_error("Using virtual evolve_state of oscillator");
            
            };
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
class harmonic : public Oscillator{
    public:
        float k;
        static osc_state none_state;
        harmonic(const ParaMap & params);
        void evolve_state(const osc_state & state, osc_state & dxdt, double t) override;
};

/**
 * @class OscillatorNetwork
 * @brief A (homogeneous) network of oscillators
 * 
 * I
*/
class OscillatorNetwork{
    public:
        OscillatorNetwork(oscillator_type osc_type, std::vector<ParaMap*> params, const Projection & self_projection);
        
        std::vector<Oscillator*> oscillators;
        
        void run(EvolutionContext * evo, double time);
        void add_oscillator(Oscillator * oscillator);
};
