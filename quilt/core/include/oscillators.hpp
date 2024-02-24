/**
 * Stuff for oscillators. First I try a sparse version.
 * I'm really aware that a dense "matrix" version of this would 
 * probably be better, but let's see.
 * 
 * Numerical methods for delay differential equations Bellen Zennaro
*/
#pragma once
#include <functional>
#include "base_objects.hpp"
#include "network.hpp"

class EvolutionContext;
class Population;

using std::vector;
using std::cout;
using std::endl;

typedef std::vector<double> osc_state;
class OscillatorNetwork;

/**Available type of oscillators*/
enum class oscillator_type : unsigned int {base_oscillator, harmonic, jensen_rit, red_wong_wang};

class ContinuousRK{
    public:
        
        // These are the coefficients of the RK method
        vector<double> a = {0, 0.5, 0.5, 1};
        vector<double> b = {1.0/3.0, 1.0/6.0, 1.0/6.0, 1.0/3.0};
        vector<double> c = {0, 0.5, 0.5, 1};

        // These two make it possible to do a sequential updating.
        // The system of equation (if no vanishing delays are present)
        // requires to update just one subsystem at a time since all the other variables
        // are locked to past values 
        osc_state proposed_state;
        vector<osc_state> proposed_evaluation;

        void set_dimension(unsigned int dimension){space_dimension = dimension;}
        void set_evolution_equation(std::function<void(const osc_state & x, osc_state & dxdt, double t)> F){evolve_state = F;};

        /**
         * The continuous parameters of the NCE. See "Natural Continuous extensions of Runge-Kutta methods", M. Zennaro, 1986.
        */
        vector<double> b_functions(double theta);

        ContinuousRK(EvolutionContext * evo):evo(evo){cout << "created CRK" << endl;};

        vector<osc_state> state_history;

        /**
         * The K coefficients of RK method.
         * 
         * For each step previously computed, there are nu intermediate steps function evalutaions.
         * For each evluation the number of coeffiecients is equal to the dimension of the oscillator.
         * Thus for a N-long history of a nu-stage RK of an M-dimensional oscillator, the K coefficients
         * have shape (N, nu, M).
        */
        vector<vector<osc_state>> evaluation_history;
        /**
         * Computes the interpolation using the Natural Continuous Extension at a given time for a given axis (one variable of interest).
        */
        double get_past(int axis, double abs_time);
        void compute_next();
        void fix_next();
    private:
        EvolutionContext * evo;
        unsigned int space_dimension = -1;
        std::function<void(const osc_state & x, osc_state & dxdt, double t)> evolve_state;

};

/**
 * @class Link
 * @brief Delay-weight link for oscillators
 * 
 * The main method is `get()`, that returns the `osc_state` of `source` at \f$t = t_{now}-\tau_{i,j} \f$.
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
        static osc_state none_state;    //!< This is temporary! The problem starts in C[-T, 0]
        oscillator_type osc_type = oscillator_type::base_oscillator;
        HierarchicalID id;
        OscillatorNetwork * oscnet;

        std::vector< Link<Oscillator, Oscillator>> incoming_osc;

        Oscillator(OscillatorNetwork * oscnet, EvolutionContext * evo);
        void connect(Oscillator * osc, float weight, float delay);
        
        std::function<void(const osc_state & x, osc_state & dxdt, double t)> evolve_state;
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
        
        void init_oscillators(vector<osc_state> init_conds);
        void run(EvolutionContext * evo, double time);
    private:
        EvolutionContext * evo;
};